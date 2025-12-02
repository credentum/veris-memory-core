"""
API Key Authentication Middleware
Sprint 13 - Phase 2: Security & Attribution

Provides simple API key authentication with role-based capabilities.
Integrates with existing RBAC system for consistent authorization.
"""

import os
import logging
import hashlib
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass
from fastapi import Request, HTTPException, Header
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class APIKeyInfo:
    """Information about an API key"""
    key_id: str
    user_id: str
    role: str
    capabilities: List[str]
    is_agent: bool  # True if this is an AI agent, False if human
    metadata: Dict[str, Any]


class APIKeyManager:
    """
    Manages API key authentication and validation.

    Sprint 13: Simple API key auth with integration to existing RBAC.
    """

    def __init__(self) -> None:
        """Initialize API key manager"""
        self.api_keys: Dict[str, APIKeyInfo] = self._load_api_keys()
        logger.info(f"API Key Manager initialized with {len(self.api_keys)} keys")

    def _load_api_keys(self) -> Dict[str, APIKeyInfo]:
        """
        Load API keys from environment variables.

        Format: API_KEY_{NAME}=key:user_id:role:is_agent
        Example: API_KEY_ADMIN=admin_key_123:admin_user:admin:false
        """
        api_keys = {}

        # Load from environment
        for env_var, env_value in os.environ.items():
            if env_var.startswith("API_KEY_"):
                try:
                    parts = env_value.split(":")
                    if len(parts) >= 4:
                        key, user_id, role, is_agent = parts[0], parts[1], parts[2], parts[3]

                        # Determine capabilities based on role
                        capabilities = self._get_role_capabilities(role)

                        api_keys[key] = APIKeyInfo(
                            key_id=env_var[8:].lower(),  # Remove API_KEY_ prefix
                            user_id=user_id,
                            role=role,
                            capabilities=capabilities,
                            is_agent=is_agent.lower() == "true",
                            metadata={"source": "environment"}
                        )
                        logger.info(f"Loaded API key: {env_var[8:]} (role: {role}, agent: {is_agent})")
                except Exception as e:
                    logger.error(f"Failed to parse API key {env_var}: {e}")

        # ⚠️ SECURITY WARNING: Default test key for development only
        # This test key should NEVER be used in production environments.
        # Set ENVIRONMENT=production to disable automatic test key creation.
        # Generate secure, unique API keys for all production deployments.
        if not api_keys and os.getenv("ENVIRONMENT", "production") != "production":
            default_key = "vmk_test_a1b2c3d4e5f6789012345678901234567890"
            api_keys[default_key] = APIKeyInfo(
                key_id="default_test",
                user_id="test_user",
                role="writer",
                capabilities=["store_context", "retrieve_context", "query_graph", "update_scratchpad", "get_agent_state"],
                is_agent=True,
                metadata={"source": "default", "warning": "TEST_KEY_ONLY_NEVER_USE_IN_PRODUCTION"}
            )
            logger.warning("⚠️ SECURITY: Using default test API key - NEVER USE IN PRODUCTION")

        return api_keys

    def _get_role_capabilities(self, role: str) -> List[str]:
        """Get capabilities for a role (matches RBAC definitions)"""
        role_capabilities = {
            "admin": ["*"],  # All capabilities
            "writer": ["store_context", "retrieve_context", "query_graph", "update_scratchpad", "get_agent_state"],
            "reader": ["retrieve_context", "query_graph", "get_agent_state"],
            "guest": ["retrieve_context"]
        }
        return role_capabilities.get(role, ["retrieve_context"])

    def validate_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """
        Validate an API key.

        Args:
            api_key: The API key to validate

        Returns:
            APIKeyInfo if valid, None otherwise
        """
        # Direct lookup
        if api_key in self.api_keys:
            return self.api_keys[api_key]

        # Hash lookup (for stored hashed keys)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        for stored_key, key_info in self.api_keys.items():
            stored_hash = hashlib.sha256(stored_key.encode()).hexdigest()
            if key_hash == stored_hash:
                return key_info

        return None

    def has_capability(self, api_key: str, capability: str) -> bool:
        """
        Check if API key has specific capability.

        Args:
            api_key: The API key to check
            capability: The capability to verify

        Returns:
            True if the key has the capability, False otherwise
        """
        key_info = self.validate_key(api_key)
        if not key_info:
            return False

        return "*" in key_info.capabilities or capability in key_info.capabilities

    def is_human(self, api_key: str) -> bool:
        """
        Check if API key belongs to a human (not an agent).

        Args:
            api_key: The API key to check

        Returns:
            True if the key belongs to a human, False if it's an agent or invalid
        """
        key_info = self.validate_key(api_key)
        if not key_info:
            return False

        return not key_info.is_agent


# Global API key manager instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get or create global API key manager"""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
) -> APIKeyInfo:
    """
    FastAPI dependency to verify API key from headers.

    Checks both X-API-Key and Authorization: Bearer {key} headers.
    Sprint 13: Phase 2 - API Key Authentication

    Args:
        request: FastAPI request object
        x_api_key: API key from X-API-Key header
        authorization: API key from Authorization header

    Returns:
        APIKeyInfo if valid

    Raises:
        HTTPException: If API key is missing or invalid
    """
    manager = get_api_key_manager()

    # Extract API key from headers
    api_key = None

    # Check X-API-Key header first
    if x_api_key:
        api_key = x_api_key
    # Check Authorization: Bearer {key}
    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]  # Remove "Bearer " prefix

    # No API key provided
    if not api_key:
        # Allow unauthenticated access if AUTH_REQUIRED=false (development only)
        if os.getenv("AUTH_REQUIRED", "true").lower() == "false":
            logger.warning("⚠️ Unauthenticated request allowed (AUTH_REQUIRED=false)")
            return APIKeyInfo(
                key_id="unauthenticated",
                user_id="anonymous",
                role="guest",
                capabilities=["retrieve_context"],
                is_agent=True,
                metadata={"warning": "unauthenticated"}
            )

        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via X-API-Key header or Authorization: Bearer {key}"
        )

    # Validate API key
    key_info = manager.validate_key(api_key)

    if not key_info:
        logger.warning(f"Invalid API key attempted: {api_key[:20]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    logger.debug(f"Authenticated: {key_info.user_id} (role: {key_info.role})")
    return key_info


def require_capability(capability: str) -> Callable:
    """
    Decorator to require specific capability for an endpoint.
    Sprint 13: Phase 2

    Args:
        capability: The required capability (e.g., "store_context")

    Returns:
        Decorator function

    Usage:
        @app.post("/protected")
        @require_capability("store_context")
        async def protected_endpoint():
            pass
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, api_key_info: Optional[APIKeyInfo] = None, **kwargs) -> Any:
            if not api_key_info:
                raise HTTPException(status_code=401, detail="Authentication required")

            manager = get_api_key_manager()

            # Check capability
            if "*" not in api_key_info.capabilities and capability not in api_key_info.capabilities:
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing capability: {capability}"
                )

            return await func(*args, api_key_info=api_key_info, **kwargs)

        return wrapper
    return decorator


def require_human() -> Callable:
    """
    Decorator to require human user (not AI agent).
    Sprint 13: Phase 2.3 - Human-only operations

    Returns:
        Decorator function that enforces human-only access

    Usage:
        @app.delete("/context/{id}")
        @require_human()
        async def delete_context(id: str, api_key_info: APIKeyInfo):
            pass
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, api_key_info: Optional[APIKeyInfo] = None, **kwargs) -> Any:
            if not api_key_info:
                raise HTTPException(status_code=401, detail="Authentication required")

            if api_key_info.is_agent:
                raise HTTPException(
                    status_code=403,
                    detail="This operation requires human authorization. AI agents cannot perform deletions."
                )

            return await func(*args, api_key_info=api_key_info, **kwargs)

        return wrapper
    return decorator


__all__ = [
    "APIKeyManager",
    "APIKeyInfo",
    "get_api_key_manager",
    "verify_api_key",
    "require_capability",
    "require_human",
]
