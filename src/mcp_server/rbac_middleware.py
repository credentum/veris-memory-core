"""
RBAC Middleware for MCP Server
Sprint 10 - Issue 002: Integration of RBAC with MCP server
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from datetime import datetime

# Import RBAC components
from ..auth.rbac import (
    RBACManager,
    CapabilityManager,
    ServiceAuthManager,
    RateLimiter,
    AuditLogger,
    AuthResult,
    RateLimitResult
)
from ..auth.token_validator import TokenValidator, ValidationResult

logger = logging.getLogger(__name__)


class MCPRBACMiddleware:
    """
    RBAC middleware for MCP server tool authorization.
    Integrates with the MCP server to provide authentication and authorization
    for all tool calls based on JWT tokens and role-based permissions.
    """
    
    # Tool to capability mapping
    TOOL_CAPABILITIES = {
        "store_context": ["store_context"],
        "retrieve_context": ["retrieve_context"],
        "query_graph": ["query_graph"],
        "update_scratchpad": ["update_scratchpad"],
        "get_agent_state": ["get_agent_state"],
        "detect_communities": ["query_graph", "retrieve_context"],
        "select_tools": ["retrieve_context"],
        "list_available_tools": ["retrieve_context"],
        "get_tool_info": ["retrieve_context"],
    }
    
    # Tools that require elevated permissions
    ADMIN_TOOLS = []  # Currently no admin-only tools
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RBAC middleware.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.rbac_manager = RBACManager()
        self.capability_manager = CapabilityManager()
        self.service_auth_manager = ServiceAuthManager()
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger()
        self.token_validator = TokenValidator()
        
        # Configuration flags
        self.enabled = self._get_config("rbac_enabled", True)
        self.strict_mode = self._get_config("rbac_strict_mode", False)
        self.audit_enabled = self._get_config("audit_enabled", True)
        self.rate_limiting_enabled = self._get_config("rate_limiting_enabled", True)
        
        logger.info(f"RBAC Middleware initialized (enabled: {self.enabled}, strict: {self.strict_mode})")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to environment variable."""
        # Check config dict first
        if key in self.config:
            return self.config[key]
        
        # Check environment variable
        env_key = f"MCP_{key.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            # Convert string to boolean if needed
            if isinstance(default, bool):
                return env_value.lower() in ["true", "1", "yes", "on"]
            return type(default)(env_value)
        
        return default
    
    def extract_token(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Extract JWT token from request context.
        
        Args:
            context: Request context containing headers or auth info
            
        Returns:
            JWT token string or None if not found
        """
        # Check for token in various locations
        
        # 1. Direct token field
        if "token" in context:
            return context["token"]
        
        # 2. Authorization header
        if "headers" in context:
            auth_header = context["headers"].get("Authorization", "")
            if auth_header.startswith("Bearer "):
                return auth_header[7:]
        
        # 3. Auth field with token
        if "auth" in context and isinstance(context["auth"], dict):
            if "token" in context["auth"]:
                return context["auth"]["token"]
            if "jwt" in context["auth"]:
                return context["auth"]["jwt"]
        
        # 4. Metadata field (MCP specific)
        if "metadata" in context and isinstance(context["metadata"], dict):
            if "auth_token" in context["metadata"]:
                return context["metadata"]["auth_token"]
            if "token" in context["metadata"]:
                return context["metadata"]["token"]
        
        return None
    
    def authorize_tool(
        self,
        tool_name: str,
        token: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AuthResult:
        """
        Authorize access to a specific tool.
        
        Args:
            tool_name: Name of the tool to authorize
            token: JWT token for authentication
            context: Optional request context
            
        Returns:
            AuthResult with authorization status
        """
        # Validate token
        validation = self.token_validator.validate(token)
        
        if not validation.is_valid:
            return AuthResult(
                authorized=False,
                error=validation.error,
                status_code=401
            )
        
        # Get required capabilities for tool
        required_caps = self.TOOL_CAPABILITIES.get(tool_name, [])
        
        # Check if tool requires admin permissions
        if tool_name in self.ADMIN_TOOLS and validation.role != "admin":
            result = AuthResult(
                authorized=False,
                user_id=validation.user_id,
                role=validation.role,
                error="Admin permission required",
                status_code=403
            )
        else:
            # Check capabilities
            user_caps = set(validation.capabilities)
            has_permission = (
                validation.role == "admin" or  # Admin bypass
                "*" in user_caps or  # Wildcard permission
                any(cap in user_caps for cap in required_caps)  # Has required capability
            )
            
            if has_permission:
                result = AuthResult(
                    authorized=True,
                    user_id=validation.user_id,
                    role=validation.role,
                    capabilities=validation.capabilities,
                    status_code=200
                )
            else:
                result = AuthResult(
                    authorized=False,
                    user_id=validation.user_id,
                    role=validation.role,
                    capabilities=validation.capabilities,
                    error=f"Missing required capabilities: {required_caps}",
                    status_code=403
                )
        
        # Audit log
        if self.audit_enabled:
            self.audit_logger.log_auth_attempt(
                user_id=result.user_id or "unknown",
                operation=f"tool:{tool_name}",
                authorized=result.authorized,
                metadata={
                    "role": result.role,
                    "required_capabilities": required_caps,
                    "error": result.error
                }
            )
        
        return result
    
    def check_rate_limit(
        self,
        token: str,
        tool_name: str
    ) -> RateLimitResult:
        """
        Check rate limit for tool access.
        
        Args:
            token: JWT token
            tool_name: Name of the tool
            
        Returns:
            RateLimitResult with rate limit status
        """
        if not self.rate_limiting_enabled:
            return RateLimitResult(allowed=True, limit=0)
        
        # Check general rate limit
        result = self.rate_limiter.check_rate_limit(token)
        
        # Could add tool-specific rate limits here
        # For now, use general rate limit for all tools
        
        return result
    
    def wrap_tool(self, tool_name: str, tool_func: Callable) -> Callable:
        """
        Wrap a tool function with RBAC authorization.
        
        Args:
            tool_name: Name of the tool
            tool_func: Original tool function
            
        Returns:
            Wrapped function with RBAC checks
        """
        @wraps(tool_func)
        async def wrapped_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Wrapped tool with RBAC authorization."""
            
            # If RBAC is disabled, call original function
            if not self.enabled:
                return await tool_func(arguments)
            
            # Extract context and token
            context = arguments.get("__context__", {})
            token = self.extract_token(context)
            
            # If no token and strict mode, deny access
            if not token and self.strict_mode:
                logger.warning(f"No token provided for tool {tool_name} in strict mode")
                return {
                    "success": False,
                    "error": "Authentication required",
                    "error_type": "auth_required",
                    "status_code": 401
                }
            
            # If no token and not strict mode, allow with guest permissions
            if not token:
                logger.info(f"No token for tool {tool_name}, using guest access")
                # Create a guest token for rate limiting
                guest_token = self.capability_manager.create_token(
                    user_id="guest",
                    role="guest",
                    capabilities=["retrieve_context"],
                    expires_in=60  # Short-lived guest token
                )
                token = guest_token
            
            # Authorize tool access
            auth_result = self.authorize_tool(tool_name, token, context)
            
            if not auth_result.authorized:
                logger.warning(
                    f"Authorization failed for tool {tool_name}: {auth_result.error}"
                )
                return {
                    "success": False,
                    "error": auth_result.error,
                    "error_type": "authorization_failed",
                    "status_code": auth_result.status_code
                }
            
            # Check rate limit
            rate_limit_result = self.check_rate_limit(token, tool_name)
            
            if not rate_limit_result.allowed:
                logger.warning(
                    f"Rate limit exceeded for tool {tool_name}, retry after {rate_limit_result.retry_after}s"
                )
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "error_type": "rate_limit",
                    "retry_after": rate_limit_result.retry_after,
                    "remaining": rate_limit_result.remaining,
                    "limit": rate_limit_result.limit,
                    "status_code": 429
                }
            
            # Add auth context to arguments
            arguments["__auth__"] = {
                "user_id": auth_result.user_id,
                "role": auth_result.role,
                "capabilities": auth_result.capabilities
            }
            
            # Call original tool function
            try:
                result = await tool_func(arguments)
                
                # Add rate limit info to successful responses
                if rate_limit_result.remaining is not None:
                    result["__rate_limit__"] = {
                        "remaining": rate_limit_result.remaining,
                        "limit": rate_limit_result.limit
                    }
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": "execution_error",
                    "status_code": 500
                }
        
        return wrapped_tool
    
    def authorize_service(
        self,
        service_name: str,
        token: str
    ) -> AuthResult:
        """
        Authorize access to a backend service.
        
        Args:
            service_name: Name of the service (neo4j, qdrant, redis, etc.)
            token: JWT token
            
        Returns:
            AuthResult with authorization status
        """
        return self.service_auth_manager.authorize_service_access(token, service_name)
    
    def get_user_capabilities(self, token: str) -> List[str]:
        """
        Get capabilities for a user from their token.
        
        Args:
            token: JWT token
            
        Returns:
            List of capability names
        """
        validation = self.token_validator.validate(token)
        
        if not validation.is_valid:
            return []
        
        return validation.capabilities
    
    def create_token(
        self,
        user_id: str,
        role: str,
        capabilities: Optional[List[str]] = None,
        expires_in: int = 3600
    ) -> str:
        """
        Create a new JWT token.
        
        Args:
            user_id: User identifier
            role: User role
            capabilities: Optional list of capabilities
            expires_in: Token expiry in seconds
            
        Returns:
            JWT token string
        """
        return self.capability_manager.create_token(
            user_id=user_id,
            role=role,
            capabilities=capabilities,
            expires_in=expires_in
        )
    
    def validate_token(self, token: str) -> ValidationResult:
        """
        Validate a JWT token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            ValidationResult with token details
        """
        return self.token_validator.validate(token)
    
    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get audit logs.
        
        Args:
            user_id: Optional filter by user ID
            limit: Maximum number of logs to return
            
        Returns:
            List of audit log entries
        """
        return self.audit_logger.get_logs(user_id=user_id, limit=limit)
    
    def cleanup(self):
        """Clean up resources."""
        # Clean up old audit logs
        self.audit_logger.cleanup_old_logs()
        logger.info("RBAC middleware cleanup completed")


def create_rbac_middleware(config: Optional[Dict[str, Any]] = None) -> MCPRBACMiddleware:
    """
    Factory function to create RBAC middleware instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        MCPRBACMiddleware instance
    """
    return MCPRBACMiddleware(config)


# Export main components
__all__ = [
    "MCPRBACMiddleware",
    "create_rbac_middleware",
]