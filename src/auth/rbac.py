"""
Role-Based Access Control (RBAC) Implementation
Sprint 10 - Issue 002: RBAC & Per-Capability Scopes
"""

import os
import jwt
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import redis
from functools import wraps
from .token_validator import TokenValidator, ValidationResult

logger = logging.getLogger(__name__)


class Role(Enum):
    """System roles with hierarchical permissions"""
    ADMIN = "admin"
    WRITER = "writer"
    READER = "reader"
    GUEST = "guest"


class Capability(Enum):
    """System capabilities that can be granted"""
    STORE_CONTEXT = "store_context"
    RETRIEVE_CONTEXT = "retrieve_context"
    QUERY_GRAPH = "query_graph"
    UPDATE_SCRATCHPAD = "update_scratchpad"
    GET_AGENT_STATE = "get_agent_state"
    ADMIN_OPERATIONS = "admin_operations"


@dataclass
class RoleDefinition:
    """Definition of a role with its capabilities and limits"""
    name: str
    capabilities: Set[str]
    rate_limit: int  # requests per minute
    max_query_cost: int
    inherits_from: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthResult:
    """Result of authorization check"""
    authorized: bool
    user_id: Optional[str] = None
    role: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    error: Optional[str] = None
    status_code: int = 200
    executed: bool = False


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int = 0
    retry_after: int = 0  # seconds until retry
    limit: int = 0


class RBACManager:
    """Main RBAC manager for role and permission management"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize RBAC manager with role definitions"""
        self.roles: Dict[str, RoleDefinition] = self._load_role_definitions(config_path)
        self.role_hierarchy = self._build_role_hierarchy()
        self._redis_client = self._init_redis()
        
    def _load_role_definitions(self, config_path: Optional[str]) -> Dict[str, RoleDefinition]:
        """Load role definitions from configuration"""
        # Default role definitions
        roles = {
            "admin": RoleDefinition(
                name="admin",
                capabilities={cap.value for cap in Capability},  # All capabilities
                rate_limit=1000,  # per minute
                max_query_cost=50000,
                inherits_from=None
            ),
            "writer": RoleDefinition(
                name="writer",
                capabilities={
                    Capability.STORE_CONTEXT.value,
                    Capability.RETRIEVE_CONTEXT.value,
                    Capability.QUERY_GRAPH.value,
                    Capability.UPDATE_SCRATCHPAD.value,
                    Capability.GET_AGENT_STATE.value,
                },
                rate_limit=100,
                max_query_cost=5000,
                inherits_from="reader"
            ),
            "reader": RoleDefinition(
                name="reader",
                capabilities={
                    Capability.RETRIEVE_CONTEXT.value,
                    Capability.QUERY_GRAPH.value,
                    Capability.GET_AGENT_STATE.value,
                },
                rate_limit=60,
                max_query_cost=1000,
                inherits_from="guest"
            ),
            "guest": RoleDefinition(
                name="guest",
                capabilities={
                    Capability.RETRIEVE_CONTEXT.value,
                },
                rate_limit=10,
                max_query_cost=100,
                inherits_from=None
            ),
        }
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for role_name, role_config in config.get("roles", {}).items():
                    if role_name in roles:
                        roles[role_name].metadata.update(role_config.get("metadata", {}))
        
        return roles
    
    def _build_role_hierarchy(self) -> Dict[str, List[str]]:
        """Build role hierarchy for inheritance"""
        hierarchy = {}
        
        for role_name, role_def in self.roles.items():
            chain = [role_name]
            current = role_def
            
            while current.inherits_from:
                chain.append(current.inherits_from)
                current = self.roles.get(current.inherits_from)
                if not current:
                    break
            
            hierarchy[role_name] = chain
        
        return hierarchy
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection for rate limiting"""
        try:
            return redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                password=os.environ.get("REDIS_PASSWORD"),
                decode_responses=True,
                ssl=os.environ.get("REDIS_TLS", "false").lower() == "true"
            )
        except Exception as e:
            logger.warning(f"Redis not available for rate limiting: {e}")
            return None
    
    def role_exists(self, role: str) -> bool:
        """Check if a role exists"""
        return role in self.roles
    
    def get_role_permissions(self, role: str) -> Optional[Set[str]]:
        """Get permissions for a role"""
        if role not in self.roles:
            return None
        return self.roles[role].capabilities
    
    def get_role_hierarchy(self, role: str) -> List[str]:
        """Get role hierarchy chain"""
        return self.role_hierarchy.get(role, [role])
    
    def get_effective_permissions(self, role: str) -> Set[str]:
        """Get all effective permissions including inherited ones"""
        permissions = set()
        
        for role_name in self.get_role_hierarchy(role):
            if role_name in self.roles:
                permissions.update(self.roles[role_name].capabilities)
        
        return permissions


class CapabilityManager:
    """Manager for capability-based access control"""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize capability manager"""
        self.secret_key = secret_key or os.environ.get("JWT_SECRET", "default-secret-key")
        self.rbac_manager = RBACManager()
        
    def create_token(
        self,
        user_id: str,
        role: str,
        capabilities: Optional[List[str]] = None,
        expires_in: int = 3600  # seconds
    ) -> str:
        """Create JWT token with capabilities"""
        # If no capabilities specified, use role defaults
        if capabilities is None:
            capabilities = list(self.rbac_manager.get_effective_permissions(role))
        
        # Validate capabilities against role permissions
        role_perms = self.rbac_manager.get_effective_permissions(role)
        validated_caps = [cap for cap in capabilities if cap in role_perms]
        
        # Create token payload with required JWT claims
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "role": role,
            "capabilities": validated_caps,
            "iat": now,
            "exp": now + timedelta(seconds=expires_in),
            "iss": "context-store",  # Required issuer claim
            "aud": "context-store-api",  # Required audience claim
            "jti": hashlib.sha256(f"{user_id}{now}".encode()).hexdigest()[:16]
        }
        
        # Generate token
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        logger.info(f"Token created for user {user_id} with role {role}")
        return token
    
    def verify_capability(self, token: str, capability: str) -> bool:
        """Verify if token has specific capability"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return capability in payload.get("capabilities", [])
        except jwt.InvalidTokenError:
            return False
    
    def get_capability_limits(self, token: str, capability: str) -> Optional[Dict[str, Any]]:
        """Get rate limits for a capability"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            role = payload.get("role", "guest")
            
            if role in self.rbac_manager.roles:
                role_def = self.rbac_manager.roles[role]
                return {
                    "requests_per_minute": role_def.rate_limit,
                    "max_query_cost": role_def.max_query_cost,
                    "capability": capability,
                    "role": role
                }
            
        except jwt.InvalidTokenError:
            pass
        
        return None


class RBACMiddleware:
    """Middleware for enforcing RBAC in requests"""
    
    def __init__(self):
        """Initialize RBAC middleware"""
        self.capability_manager = CapabilityManager()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        
    def check_permission(self, token: str, operation: str) -> AuthResult:
        """Check if token has permission for operation"""
        result = AuthResult(authorized=False, executed=False)
        
        try:
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.capability_manager.secret_key,
                algorithms=["HS256"]
            )
            
            user_id = payload.get("sub")
            role = payload.get("role", "guest")
            capabilities = payload.get("capabilities", [])
            
            result.user_id = user_id
            result.role = role
            result.capabilities = capabilities
            
            # Check if operation is in capabilities
            if operation in capabilities or "*" in capabilities:
                result.authorized = True
                result.status_code = 200
            else:
                result.authorized = False
                result.status_code = 403
                result.error = "Insufficient permissions"
            
        except jwt.ExpiredSignatureError:
            result.status_code = 401
            result.error = "Token expired"
        except jwt.InvalidTokenError:
            result.status_code = 401
            result.error = "Invalid token"
        except Exception as e:
            result.status_code = 500
            result.error = str(e)
        
        # Log the authorization attempt
        self.audit_logger.log_auth_attempt(
            user_id=result.user_id or "unknown",
            operation=operation,
            authorized=result.authorized,
            metadata={"role": result.role, "error": result.error}
        )
        
        return result
    
    def require_capability(self, capability: str):
        """Decorator to require specific capability"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract token from request context
                token = self._extract_token()
                
                if not token:
                    return AuthResult(
                        authorized=False,
                        status_code=401,
                        error="No token provided"
                    )
                
                # Check permission
                result = self.check_permission(token, capability)
                
                if not result.authorized:
                    return result
                
                # Execute function if authorized
                result.executed = True
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def _extract_token(self) -> Optional[str]:
        """Extract token from request (implement based on framework)"""
        # This would be implemented based on the web framework
        # For example, from Flask request headers
        return None



class ServiceAuthManager:
    """Manager for cross-service authorization"""
    
    def __init__(self):
        """Initialize service auth manager"""
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self.token_validator = TokenValidator()
        
        # Service access requirements
        self.service_requirements = {
            "neo4j": ["store_context", "query_graph"],
            "qdrant": ["store_context", "retrieve_context"],
            "redis": ["retrieve_context"],
            "mcp_server": ["retrieve_context"]
        }
    
    def authorize_service_access(self, token: str, service: str) -> AuthResult:
        """Authorize access to a specific service"""
        # Validate token
        validation_result = self.token_validator.validate(token)
        
        if not validation_result.authorized:
            return validation_result
        
        # Check service requirements
        required_caps = self.service_requirements.get(service, [])
        user_caps = set(validation_result.capabilities)
        
        # Admin bypass
        if validation_result.role == "admin" or "*" in user_caps:
            authorized = True
        else:
            # Check if user has any required capability
            authorized = any(cap in user_caps for cap in required_caps)
        
        result = AuthResult(
            authorized=authorized,
            user_id=validation_result.user_id,
            role=validation_result.role,
            capabilities=validation_result.capabilities,
            status_code=200 if authorized else 403
        )
        
        # Log access attempt
        self.audit_logger.log_auth_attempt(
            user_id=result.user_id,
            operation=f"access_{service}",
            authorized=authorized,
            metadata={"service": service, "role": result.role}
        )
        
        return result
    
    def get_audit_logs(self, service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit logs for service access"""
        return self.audit_logger.get_logs(service=service)


class RateLimiter:
    """Rate limiting based on role and capability"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, secret_key: Optional[str] = None):
        """Initialize rate limiter"""
        self.redis_client = redis_client or self._init_redis()
        self.rbac_manager = RBACManager()
        self.token_validator = TokenValidator(secret_key=secret_key)
        
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection"""
        try:
            return redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                password=os.environ.get("REDIS_PASSWORD"),
                decode_responses=True
            )
        except Exception:
            return None
    
    def check_rate_limit(self, token: str) -> RateLimitResult:
        """Check rate limit for token"""
        # Validate token
        validation = self.token_validator.validate(token)
        if not validation.authorized:
            return RateLimitResult(allowed=False)
        
        role = validation.role or "guest"
        user_id = validation.user_id
        
        # Get rate limit for role
        role_def = self.rbac_manager.roles.get(role)
        if not role_def:
            return RateLimitResult(allowed=False)
        
        limit = role_def.rate_limit
        
        # Check rate limit using sliding window
        if self.redis_client:
            try:
                key = f"rate_limit:{user_id}:{role}"
                current_time = time.time()
                window_start = current_time - 60  # 1 minute window
                
                # Remove old entries
                self.redis_client.zremrangebyscore(key, 0, window_start)
                
                # Count requests in window
                request_count = self.redis_client.zcard(key)
                
                if request_count >= limit:
                    # Calculate retry after
                    oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
                    if oldest:
                        retry_after = int(60 - (current_time - oldest[0][1]))
                    else:
                        retry_after = 60
                    
                    return RateLimitResult(
                        allowed=False,
                        remaining=0,
                        retry_after=retry_after,
                        limit=limit
                    )
                
                # Add current request
                self.redis_client.zadd(key, {str(current_time): current_time})
                self.redis_client.expire(key, 60)
                
                return RateLimitResult(
                    allowed=True,
                    remaining=limit - request_count - 1,
                    retry_after=0,
                    limit=limit
                )
            except Exception as e:
                logger.warning(f"Redis rate limiting failed, allowing request: {e}")
                # Fall through to Redis-free mode
        
        # Fallback if Redis not available (allow all requests)
        return RateLimitResult(allowed=True, limit=limit, remaining=limit-1)
    
    def check_capability_limit(self, token: str, capability: str) -> RateLimitResult:
        """Check rate limit for specific capability"""
        # Similar to check_rate_limit but with capability-specific limits
        return self.check_rate_limit(token)  # Simplified for now


class AuditLogger:
    """Audit logging for RBAC operations"""
    
    def __init__(self, retention_days: int = 30):
        """Initialize audit logger"""
        self.retention_days = retention_days
        self.logs: List[Dict[str, Any]] = []
        
    def log_auth_attempt(
        self,
        user_id: str,
        operation: str,
        authorized: bool,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Log an authorization attempt"""
        log_entry = {
            "timestamp": timestamp or datetime.now(timezone.utc),
            "user_id": user_id,
            "operation": operation,
            "authorized": authorized,
            "metadata": metadata or {}
        }
        
        self.logs.append(log_entry)
        
        # Log to system logger
        if authorized:
            logger.info(f"Auth granted: {user_id} -> {operation}")
        else:
            logger.warning(f"Auth denied: {user_id} -> {operation}")
    
    def get_logs(
        self,
        user_id: Optional[str] = None,
        service: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve audit logs"""
        filtered_logs = self.logs
        
        if user_id:
            filtered_logs = [l for l in filtered_logs if l["user_id"] == user_id]
        
        if service:
            filtered_logs = [
                l for l in filtered_logs
                if service in l.get("metadata", {}).get("service", "")
            ]
        
        return filtered_logs[-limit:]
    
    def cleanup_old_logs(self):
        """Remove logs older than retention period"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        self.logs = [l for l in self.logs if l["timestamp"] > cutoff]


class SessionManager:
    """Session management with RBAC"""
    
    def __init__(self):
        """Initialize session manager"""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.token_validator = TokenValidator()
        
    def create_session(self, token: str) -> str:
        """Create session from token"""
        import uuid
        
        validation = self.token_validator.validate(token)
        if not validation.authorized:
            raise ValueError("Invalid token")
        
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": validation.user_id,
            "role": validation.role,
            "capabilities": validation.capabilities,
            "created_at": datetime.now(timezone.utc),
            "token": token
        }
        
        return session_id
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if session is active"""
        return session_id in self.sessions
    
    def get_session_permissions(self, session_id: str) -> List[str]:
        """Get permissions for session"""
        if session_id in self.sessions:
            return self.sessions[session_id].get("capabilities", [])
        return []
    
    def expire_session(self, session_id: str):
        """Expire a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


class TokenManager:
    """Token lifecycle management"""
    
    def __init__(self):
        """Initialize token manager"""
        self.capability_manager = CapabilityManager()
        self.revoked_tokens: Set[str] = set()
        
    def create_token(
        self,
        user_id: str,
        role: str,
        capabilities: Optional[List[str]] = None
    ) -> str:
        """Create new token"""
        return self.capability_manager.create_token(user_id, role, capabilities)
    
    def refresh_token(self, old_token: str) -> str:
        """Refresh token before expiry"""
        # Validate old token
        try:
            payload = jwt.decode(
                old_token,
                self.capability_manager.secret_key,
                algorithms=["HS256"]
            )
            
            # Revoke old token
            self.revoked_tokens.add(old_token)
            
            # Create new token with same claims
            return self.create_token(
                user_id=payload["sub"],
                role=payload["role"],
                capabilities=payload.get("capabilities")
            )
            
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token for refresh")
    
    def validate_token(self, token: str) -> AuthResult:
        """Validate token and check if revoked"""
        if token in self.revoked_tokens:
            return AuthResult(
                authorized=False,
                error="Token has been revoked"
            )
        
        validator = TokenValidator()
        return validator.validate(token)


# Export main components
__all__ = [
    "RBACManager",
    "CapabilityManager",
    "RBACMiddleware",
    "TokenValidator",
    "ServiceAuthManager",
    "RateLimiter",
    "AuditLogger",
    "SessionManager",
    "TokenManager",
    "Role",
    "Capability",
    "AuthResult",
    "RateLimitResult",
]