"""
Scope validation middleware for fact storage and retrieval operations.

This module enforces strict namespace/tenant/user scoping on all fact operations
to prevent cross-user data leakage and ensure proper isolation.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from functools import wraps
import logging

try:
    from ..core.agent_namespace import AgentNamespace, NamespaceError
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.agent_namespace import AgentNamespace, NamespaceError

logger = logging.getLogger(__name__)


@dataclass
class ScopeContext:
    """Represents the scope context for a request."""
    namespace: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if scope context has minimum required fields."""
        return bool(self.namespace and self.user_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "namespace": self.namespace,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id
        }


class ScopeValidationError(Exception):
    """Raised when scope validation fails."""
    
    def __init__(self, message: str, error_type: str = "scope_validation_error", scope_context: Optional[ScopeContext] = None):
        super().__init__(message)
        self.error_type = error_type
        self.scope_context = scope_context


class ScopeValidator:
    """
    Enforces namespace/tenant/user scoping on all operations.
    
    This validator ensures that fact storage and retrieval operations
    are properly scoped to prevent cross-user data leakage.
    """

    def __init__(self, agent_namespace: Optional[AgentNamespace] = None):
        self.agent_namespace = agent_namespace or AgentNamespace()
        
        # Required fields for different operation types
        self.fact_operation_requirements = {
            "store_fact": ["namespace", "user_id"],
            "retrieve_fact": ["namespace", "user_id"],
            "delete_fact": ["namespace", "user_id"],
            "list_facts": ["namespace", "user_id"],
            "store_context": ["namespace"],  # Context can be agent-only
            "retrieve_context": ["namespace"]
        }

    def extract_scope_context(self, request_data: Dict[str, Any]) -> ScopeContext:
        """
        Extract scope context from request data.
        
        Args:
            request_data: Request parameters/data
            
        Returns:
            ScopeContext extracted from request
            
        Raises:
            ScopeValidationError: If required scope fields are missing
        """
        # Try different common field names for each scope component
        namespace = (request_data.get("namespace") or 
                    request_data.get("agent_id") or
                    request_data.get("agent"))
        
        tenant_id = (request_data.get("tenant_id") or
                    request_data.get("tenant") or
                    request_data.get("organization_id"))
        
        user_id = (request_data.get("user_id") or
                  request_data.get("user") or
                  request_data.get("user_identifier"))
        
        agent_id = (request_data.get("agent_id") or
                   request_data.get("agent"))
        
        session_id = (request_data.get("session_id") or
                     request_data.get("session"))

        scope = ScopeContext(
            namespace=namespace,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id
        )

        logger.debug(f"Extracted scope context: {scope.to_dict()}")
        return scope

    def validate_scope(self, operation: str, scope: ScopeContext) -> None:
        """
        Validate scope context for an operation.
        
        Args:
            operation: Type of operation (store_fact, retrieve_fact, etc.)
            scope: Scope context to validate
            
        Raises:
            ScopeValidationError: If validation fails
        """
        if not scope:
            raise ScopeValidationError(
                "Scope context is required",
                error_type="missing_scope",
                scope_context=scope
            )

        # Get requirements for this operation
        requirements = self.fact_operation_requirements.get(operation, ["namespace", "user_id"])
        
        # Check required fields
        missing_fields = []
        for field in requirements:
            if not getattr(scope, field, None):
                missing_fields.append(field)

        if missing_fields:
            raise ScopeValidationError(
                f"Missing required scope fields for {operation}: {missing_fields}",
                error_type="missing_required_fields",
                scope_context=scope
            )

        # Validate namespace format if present
        if scope.namespace and not self.agent_namespace.validate_agent_id(scope.namespace):
            raise ScopeValidationError(
                f"Invalid namespace format: {scope.namespace}",
                error_type="invalid_namespace",
                scope_context=scope
            )

        # Validate user_id format if present (basic validation)
        if scope.user_id and (len(scope.user_id) < 1 or len(scope.user_id) > 128):
            raise ScopeValidationError(
                f"Invalid user_id format: {scope.user_id}",
                error_type="invalid_user_id",
                scope_context=scope
            )

        logger.debug(f"Scope validation passed for operation {operation}")

    def apply_scope_filter(self, query_params: Dict[str, Any], scope: ScopeContext) -> Dict[str, Any]:
        """
        Apply scope filters to query parameters.
        
        Args:
            query_params: Original query parameters
            scope: Scope context to apply
            
        Returns:
            Modified query parameters with scope filters applied
        """
        filtered_params = query_params.copy()
        
        # Always enforce namespace filtering
        if scope.namespace:
            filtered_params["namespace"] = scope.namespace
        
        # Enforce user filtering for user-scoped operations
        if scope.user_id:
            filtered_params["user_id"] = scope.user_id
        
        # Add tenant filtering if available
        if scope.tenant_id:
            filtered_params["tenant_id"] = scope.tenant_id

        # Add session context if available
        if scope.session_id:
            filtered_params["session_id"] = scope.session_id

        logger.debug(f"Applied scope filters: {filtered_params}")
        return filtered_params

    def check_cross_scope_access(self, requesting_scope: ScopeContext, target_scope: ScopeContext) -> bool:
        """
        Check if one scope can access data from another scope.
        
        Args:
            requesting_scope: Scope making the request
            target_scope: Scope of the target data
            
        Returns:
            True if access is allowed, False otherwise
        """
        # Same namespace check
        if requesting_scope.namespace != target_scope.namespace:
            logger.warning(f"Cross-namespace access attempted: {requesting_scope.namespace} -> {target_scope.namespace}")
            return False

        # Same user check for user-scoped data
        if target_scope.user_id and requesting_scope.user_id != target_scope.user_id:
            logger.warning(f"Cross-user access attempted: {requesting_scope.user_id} -> {target_scope.user_id}")
            return False

        # Same tenant check if both have tenant info
        if (target_scope.tenant_id and requesting_scope.tenant_id and 
            requesting_scope.tenant_id != target_scope.tenant_id):
            logger.warning(f"Cross-tenant access attempted: {requesting_scope.tenant_id} -> {target_scope.tenant_id}")
            return False

        return True

    def create_scoped_key(self, scope: ScopeContext, operation: str, additional_key: str = "") -> str:
        """
        Create a scoped key for storage operations.
        
        Args:
            scope: Scope context
            operation: Operation type
            additional_key: Additional key component
            
        Returns:
            Scoped storage key
        """
        key_parts = [scope.namespace]
        
        if scope.user_id:
            key_parts.append(scope.user_id)
        
        if scope.tenant_id:
            key_parts.append(scope.tenant_id)
        
        key_parts.append(operation)
        
        if additional_key:
            key_parts.append(additional_key)

        scoped_key = ":".join(key_parts)
        logger.debug(f"Created scoped key: {scoped_key}")
        return scoped_key

    def validate_fact_access(self, scope: ScopeContext, fact_namespace: str, fact_user_id: str) -> bool:
        """
        Validate access to a specific fact.
        
        Args:
            scope: Requesting scope
            fact_namespace: Namespace of the fact
            fact_user_id: User ID of the fact
            
        Returns:
            True if access is allowed
        """
        if scope.namespace != fact_namespace:
            logger.warning(f"Fact access denied: namespace mismatch {scope.namespace} != {fact_namespace}")
            return False

        if scope.user_id != fact_user_id:
            logger.warning(f"Fact access denied: user mismatch {scope.user_id} != {fact_user_id}")
            return False

        return True

    def log_scope_violation(self, operation: str, scope: ScopeContext, violation_type: str, details: str = ""):
        """
        Log a scope violation for security monitoring.
        
        Args:
            operation: Operation that was attempted
            scope: Scope context of the request
            violation_type: Type of violation
            details: Additional details
        """
        violation_data = {
            "operation": operation,
            "scope": scope.to_dict() if scope else {},
            "violation_type": violation_type,
            "details": details,
            "timestamp": None  # Would use datetime.utcnow() in real implementation
        }
        
        logger.error(f"SCOPE_VIOLATION: {violation_data}")
        
        # In production, this would also:
        # - Send to security monitoring system
        # - Trigger alerts for repeated violations
        # - Update security metrics


def require_scope(*required_fields):
    """
    Decorator to enforce scope validation on functions.
    
    Args:
        required_fields: List of required scope fields
        
    Usage:
        @require_scope("namespace", "user_id")
        def store_fact(fact_data, **kwargs):
            # Function will only execute if scope is valid
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract scope from kwargs or first positional arg
            scope_data = kwargs.get('scope') or (args[0] if args else {})
            
            validator = ScopeValidator()
            scope = validator.extract_scope_context(scope_data)
            
            # Check required fields
            missing_fields = []
            for field in required_fields:
                if not getattr(scope, field, None):
                    missing_fields.append(field)
            
            if missing_fields:
                raise ScopeValidationError(
                    f"Missing required scope fields: {missing_fields}",
                    error_type="missing_required_fields",
                    scope_context=scope
                )
            
            # Add validated scope to kwargs
            kwargs['validated_scope'] = scope
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ScopeMiddleware:
    """
    Middleware class for integrating scope validation into request processing.
    
    This can be used with web frameworks or MCP servers to automatically
    validate scope on incoming requests.
    """

    def __init__(self, validator: Optional[ScopeValidator] = None):
        self.validator = validator or ScopeValidator()

    def process_request(self, operation: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming request with scope validation.
        
        Args:
            operation: Operation being performed
            request_data: Request data/parameters
            
        Returns:
            Processed request data with validated scope
            
        Raises:
            ScopeValidationError: If scope validation fails
        """
        try:
            # Extract and validate scope
            scope = self.validator.extract_scope_context(request_data)
            self.validator.validate_scope(operation, scope)
            
            # Apply scope filters
            filtered_data = self.validator.apply_scope_filter(request_data, scope)
            
            # Add validated scope to the request
            filtered_data["_validated_scope"] = scope
            
            logger.info(f"Request processed successfully: {operation}")
            return filtered_data
            
        except ScopeValidationError as e:
            self.validator.log_scope_violation(operation, e.scope_context, e.error_type, str(e))
            raise
        except Exception as e:
            logger.error(f"Unexpected error in scope validation: {e}")
            raise ScopeValidationError(
                "Internal scope validation error",
                error_type="internal_error"
            )