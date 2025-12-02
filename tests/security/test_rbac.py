"""
Test suite for RBAC & Per-Capability Scopes
Sprint 10 - Issue 002: SEC-102
"""

import pytest
import jwt
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class TestRBACRoleDefinitions:
    """Test ID: SEC-102 - RBAC Role Definition Tests"""
    
    def test_rbac_role_definitions(self):
        """Verify all roles are properly defined"""
        from src.auth.rbac import RBACManager
        
        rbac = RBACManager()
        
        # Expected roles
        expected_roles = ["admin", "writer", "reader", "guest"]
        
        for role in expected_roles:
            # Verify role exists
            assert rbac.role_exists(role), f"Role {role} not defined"
            
            # Verify permissions are correctly assigned
            permissions = rbac.get_role_permissions(role)
            assert permissions is not None, f"No permissions for role {role}"
            
            # Verify role hierarchy is enforced
            if role == "admin":
                assert "admin" in rbac.get_role_hierarchy(role)
                assert len(permissions) > 0, "Admin should have all permissions"
            elif role == "writer":
                assert "writer" in rbac.get_role_hierarchy(role)
                assert "store_context" in permissions
                assert "retrieve_context" in permissions
            elif role == "reader":
                assert "reader" in rbac.get_role_hierarchy(role)
                assert "retrieve_context" in permissions
                assert "store_context" not in permissions
            elif role == "guest":
                assert "guest" in rbac.get_role_hierarchy(role)
                assert len(permissions) <= 1, "Guest should have minimal permissions"


class TestCapabilityScoping:
    """Test capability-based access control"""
    
    def test_capability_based_scoping(self):
        """Test per-capability access control"""
        from src.auth.rbac import CapabilityManager
        
        manager = CapabilityManager()
        
        capabilities = ["store_context", "retrieve_context", "query_graph", "update_scratchpad"]
        
        # Test with token having only 'retrieve_context'
        read_only_token = manager.create_token(
            user_id="test_user",
            role="reader",
            capabilities=["retrieve_context"]
        )
        
        # Verify can read but not write
        assert manager.verify_capability(read_only_token, "retrieve_context")
        assert not manager.verify_capability(read_only_token, "store_context")
        assert not manager.verify_capability(read_only_token, "update_scratchpad")
        
        # Test with token having 'store_context'
        writer_token = manager.create_token(
            user_id="test_user",
            role="writer",
            capabilities=["store_context", "retrieve_context"]
        )
        
        # Verify can write but respects limits
        assert manager.verify_capability(writer_token, "store_context")
        assert manager.verify_capability(writer_token, "retrieve_context")
        
        # Test rate limits per capability
        rate_limits = manager.get_capability_limits(writer_token, "store_context")
        assert rate_limits is not None
        assert "requests_per_minute" in rate_limits


class TestUnauthorizedAccessPrevention:
    """Test unauthorized access is properly blocked"""
    
    def test_unauthorized_access_blocked(self):
        """Verify unauthorized operations are blocked"""
        from src.auth.rbac import RBACMiddleware
        
        middleware = RBACMiddleware()
        
        test_cases = [
            ("guest_token", "store_context", 403),
            ("reader_token", "update_scratchpad", 403),
            ("expired_token", "retrieve_context", 401),
            ("invalid_token", "query_graph", 401),
        ]
        
        for token_type, operation, expected_code in test_cases:
            # Create appropriate token
            if token_type == "guest_token":
                token = self._create_token("guest", ["retrieve_context"])
            elif token_type == "reader_token":
                token = self._create_token("reader", ["retrieve_context", "query_graph"])
            elif token_type == "expired_token":
                token = self._create_expired_token()
            elif token_type == "invalid_token":
                token = "invalid.token.here"
            
            # Attempt operation with token
            result = middleware.check_permission(token, operation)
            
            # Verify correct error code returned
            assert result.status_code == expected_code, \
                f"Expected {expected_code} for {token_type} accessing {operation}"
            
            # Verify operation did not execute
            assert result.executed is False
    
    def _create_token(self, role: str, capabilities: List[str]) -> str:
        """Helper to create test tokens"""
        payload = {
            "sub": "test_user",
            "role": role,
            "capabilities": capabilities,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "context-store",
            "aud": "context-store-api"
        }
        return jwt.encode(payload, "test_secret", algorithm="HS256")
    
    def _create_expired_token(self) -> str:
        """Create an expired token"""
        payload = {
            "sub": "test_user",
            "role": "reader",
            "capabilities": ["retrieve_context"],
            "exp": datetime.utcnow() - timedelta(hours=1),  # Expired
            "iat": datetime.utcnow() - timedelta(hours=2),
            "iss": "context-store",
            "aud": "context-store-api"
        }
        return jwt.encode(payload, "test_secret", algorithm="HS256")


class TestTokenValidation:
    """Test token validation and expiry"""
    
    def test_token_validation(self):
        """Test token validation and expiry"""
        from src.auth.token_validator import TokenValidator
        
        validator = TokenValidator(secret_key="test_secret")
        
        # Test valid token passes
        valid_token = self._create_valid_token()
        result = validator.validate(valid_token)
        assert result.is_valid is True
        assert result.error is None
        
        # Test expired token fails
        expired_token = self._create_expired_token()
        result = validator.validate(expired_token)
        assert result.is_valid is False
        assert "expired" in result.error.lower()
        
        # Test malformed token fails
        malformed_token = "not.a.valid.token"
        result = validator.validate(malformed_token)
        assert result.is_valid is False
        assert result.error is not None
        
        # Test token with invalid signature fails
        invalid_sig_token = self._create_token_with_invalid_signature()
        result = validator.validate(invalid_sig_token)
        assert result.is_valid is False
        assert "signature" in result.error.lower()
    
    def _create_valid_token(self) -> str:
        """Create a valid token"""
        payload = {
            "sub": "test_user",
            "role": "writer",
            "capabilities": ["store_context", "retrieve_context"],
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "context-store",
            "aud": "context-store-api"
        }
        return jwt.encode(payload, "test_secret", algorithm="HS256")
    
    def _create_expired_token(self) -> str:
        """Create an expired token"""
        payload = {
            "sub": "test_user",
            "role": "reader",
            "exp": datetime.utcnow() - timedelta(hours=1),
            "iat": datetime.utcnow() - timedelta(hours=2)
        }
        return jwt.encode(payload, "test_secret", algorithm="HS256")
    
    def _create_token_with_invalid_signature(self) -> str:
        """Create token with wrong secret"""
        payload = {
            "sub": "test_user",
            "role": "writer",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "context-store",
            "aud": "context-store-api"
        }
        return jwt.encode(payload, "wrong_secret", algorithm="HS256")


class TestCrossServiceAuthorization:
    """Test RBAC works across all services"""
    
    def test_cross_service_authorization(self):
        """Verify RBAC works across all services"""
        from src.auth.rbac import ServiceAuthManager
        
        auth_manager = ServiceAuthManager()
        
        services = ["mcp_server", "neo4j", "qdrant", "redis"]
        
        for service in services:
            # Test authorized access succeeds
            admin_token = self._create_admin_token()
            result = auth_manager.authorize_service_access(admin_token, service)
            assert result.authorized is True, f"Admin should access {service}"
            
            # Test unauthorized access fails
            guest_token = self._create_guest_token()
            result = auth_manager.authorize_service_access(guest_token, service)
            
            if service in ["neo4j", "qdrant"]:  # Write services
                assert result.authorized is False, f"Guest should not access {service}"
            
            # Verify audit logs are created
            audit_logs = auth_manager.get_audit_logs(service)
            assert len(audit_logs) > 0, f"No audit logs for {service}"
            
            # Verify logs contain necessary information
            latest_log = audit_logs[-1]
            assert "user_id" in latest_log
            assert "service" in latest_log
            assert "timestamp" in latest_log
            assert "authorized" in latest_log
    
    def _create_admin_token(self) -> str:
        """Create admin token"""
        payload = {
            "sub": "admin_user",
            "role": "admin",
            "capabilities": ["*"],  # All capabilities
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "context-store",
            "aud": "context-store-api"
        }
        return jwt.encode(payload, "test_secret", algorithm="HS256")
    
    def _create_guest_token(self) -> str:
        """Create guest token"""
        payload = {
            "sub": "guest_user",
            "role": "guest",
            "capabilities": ["retrieve_context"],
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "context-store",
            "aud": "context-store-api"
        }
        return jwt.encode(payload, "test_secret", algorithm="HS256")


class TestRateLimiting:
    """Test rate limiting per role/capability"""
    
    def test_role_based_rate_limiting(self):
        """Test rate limits are enforced per role"""
        from src.auth.rbac import RateLimiter
        
        limiter = RateLimiter(secret_key="test_secret")
        
        # Test guest rate limit (10/minute)
        guest_token = self._create_token("guest", ["retrieve_context"])
        
        for i in range(10):
            result = limiter.check_rate_limit(guest_token)
            assert result.allowed is True, f"Request {i+1} should be allowed"
        
        # 11th request should be blocked
        result = limiter.check_rate_limit(guest_token)
        assert result.allowed is False, "Should block after rate limit"
        assert result.retry_after > 0
        
        # Test writer rate limit (100/minute)
        writer_token = self._create_token("writer", ["store_context"])
        
        for i in range(100):
            result = limiter.check_rate_limit(writer_token)
            assert result.allowed is True
        
        # 101st request should be blocked
        result = limiter.check_rate_limit(writer_token)
        assert result.allowed is False
    
    def test_capability_based_rate_limiting(self):
        """Test rate limits per capability"""
        from src.auth.rbac import RateLimiter
        
        limiter = RateLimiter(secret_key="test_secret")
        
        token = self._create_token("writer", ["store_context", "retrieve_context"])
        
        # Different limits for different capabilities
        for _ in range(50):
            result = limiter.check_capability_limit(token, "store_context")
            assert result.allowed is True
        
        # Store might have lower limit than retrieve
        for _ in range(100):
            result = limiter.check_capability_limit(token, "retrieve_context")
            assert result.allowed is True
    
    def _create_token(self, role: str, capabilities: List[str]) -> str:
        """Helper to create test tokens"""
        payload = {
            "sub": f"{role}_user",
            "role": role,
            "capabilities": capabilities,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "context-store",
            "aud": "context-store-api"
        }
        return jwt.encode(payload, "test_secret", algorithm="HS256")


class TestAuditLogging:
    """Test audit logging for RBAC operations"""
    
    def test_audit_log_creation(self):
        """Verify audit logs are created for all auth operations"""
        from src.auth.rbac import AuditLogger
        
        logger = AuditLogger()
        
        # Test successful auth
        logger.log_auth_attempt(
            user_id="test_user",
            operation="store_context",
            authorized=True,
            metadata={"ip": "127.0.0.1"}
        )
        
        # Test failed auth
        logger.log_auth_attempt(
            user_id="test_user",
            operation="admin_operation",
            authorized=False,
            metadata={"reason": "insufficient_permissions"}
        )
        
        # Retrieve logs
        logs = logger.get_logs(user_id="test_user")
        assert len(logs) >= 2
        
        # Verify log structure
        for log in logs:
            assert "timestamp" in log
            assert "user_id" in log
            assert "operation" in log
            assert "authorized" in log
            assert "metadata" in log
    
    def test_audit_log_retention(self):
        """Test audit logs are retained properly"""
        from src.auth.rbac import AuditLogger
        
        logger = AuditLogger(retention_days=30)
        
        # Create old log (should be retained)
        old_timestamp = datetime.utcnow() - timedelta(days=29)
        logger.log_auth_attempt(
            user_id="test_user",
            operation="test_op",
            authorized=True,
            timestamp=old_timestamp
        )
        
        # Create very old log (should be purged)
        very_old_timestamp = datetime.utcnow() - timedelta(days=31)
        logger.log_auth_attempt(
            user_id="test_user",
            operation="old_op",
            authorized=True,
            timestamp=very_old_timestamp
        )
        
        # Run retention cleanup
        logger.cleanup_old_logs()
        
        # Verify only recent logs remain
        logs = logger.get_logs(user_id="test_user")
        for log in logs:
            log_age = datetime.utcnow() - log["timestamp"]
            assert log_age.days <= 30, "Old logs should be purged"


class TestPermissionInheritance:
    """Test role hierarchy and permission inheritance"""
    
    def test_role_hierarchy(self):
        """Test permission inheritance in role hierarchy"""
        from src.auth.rbac import RBACManager
        
        rbac = RBACManager()
        
        # Admin inherits all permissions
        admin_perms = rbac.get_effective_permissions("admin")
        writer_perms = rbac.get_effective_permissions("writer")
        reader_perms = rbac.get_effective_permissions("reader")
        guest_perms = rbac.get_effective_permissions("guest")
        
        # Verify hierarchy: admin > writer > reader > guest
        assert len(admin_perms) > len(writer_perms)
        assert len(writer_perms) > len(reader_perms)
        assert len(reader_perms) >= len(guest_perms)
        
        # Verify permission inheritance
        for perm in guest_perms:
            assert perm in reader_perms, f"Reader should have guest permission: {perm}"
        
        for perm in reader_perms:
            assert perm in writer_perms, f"Writer should have reader permission: {perm}"
        
        for perm in writer_perms:
            assert perm in admin_perms, f"Admin should have writer permission: {perm}"


class TestSecurityIntegration:
    """Integration tests for RBAC with other security features"""
    
    def test_rbac_with_tls(self):
        """Test RBAC works with TLS connections"""
        # This would test that RBAC properly validates tokens
        # even when connections are over TLS/mTLS
        pass
    
    def test_rbac_token_rotation(self):
        """Test token refresh and rotation"""
        from src.auth.rbac import TokenManager
        
        manager = TokenManager()
        
        # Create initial token
        initial_token = manager.create_token(
            user_id="test_user",
            role="writer",
            capabilities=["store_context"]
        )
        
        # Refresh token before expiry
        refreshed_token = manager.refresh_token(initial_token)
        
        # Verify new token is different but valid
        assert refreshed_token != initial_token
        assert manager.validate_token(refreshed_token).is_valid
        
        # Verify old token is revoked
        assert not manager.validate_token(initial_token).is_valid
    
    def test_rbac_with_session_management(self):
        """Test RBAC with session tracking"""
        from src.auth.rbac import SessionManager
        
        session_mgr = SessionManager()
        
        # Create session with token
        token = self._create_token("writer", ["store_context"])
        session_id = session_mgr.create_session(token)
        
        # Verify session is active
        assert session_mgr.is_session_active(session_id)
        
        # Verify session has correct permissions
        perms = session_mgr.get_session_permissions(session_id)
        assert "store_context" in perms
        
        # Test session expiry
        session_mgr.expire_session(session_id)
        assert not session_mgr.is_session_active(session_id)
    
    def _create_token(self, role: str, capabilities: List[str]) -> str:
        """Helper to create test tokens"""
        payload = {
            "sub": "test_user",
            "role": role,
            "capabilities": capabilities,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "context-store",
            "aud": "context-store-api"
        }
        return jwt.encode(payload, "test_secret", algorithm="HS256")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])