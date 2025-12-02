"""
Namespace Management System
Sprint 13 Phase 4.1

Path-based namespaces for organizing contexts with TTL-based locks.
Supports: /global/, /team/, /user/, /project/
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NamespaceConfig:
    """Configuration for a namespace"""
    name: str
    path: str
    default_ttl: int  # seconds
    max_size: Optional[int] = None  # max contexts in namespace
    owner: Optional[str] = None
    permissions: List[str] = None  # list of allowed operations
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = ["read", "write", "delete"]
        if self.metadata is None:
            self.metadata = {}


class NamespaceManager:
    """
    Manages hierarchical namespaces for context organization.
    Sprint 13 Phase 4.1

    Namespace Format:
    - /global/{name} - Global shared contexts
    - /team/{team_id}/{name} - Team-specific contexts
    - /user/{user_id}/{name} - User-specific contexts
    - /project/{project_id}/{name} - Project-specific contexts
    """

    # Predefined namespace types
    NAMESPACE_TYPES = {
        "global": {
            "pattern": "/global/{name}",
            "default_ttl": 2592000,  # 30 days
            "description": "Globally shared contexts accessible to all users"
        },
        "team": {
            "pattern": "/team/{team_id}/{name}",
            "default_ttl": 604800,  # 7 days
            "description": "Team-specific contexts shared within a team"
        },
        "user": {
            "pattern": "/user/{user_id}/{name}",
            "default_ttl": 86400,  # 1 day
            "description": "User-private contexts accessible only to the owner"
        },
        "project": {
            "pattern": "/project/{project_id}/{name}",
            "default_ttl": 1209600,  # 14 days
            "description": "Project-specific contexts for project collaboration"
        }
    }

    def __init__(self, redis_client=None):
        """Initialize namespace manager"""
        self.redis_client = redis_client
        self.lock_ttl = 30  # seconds for locks
        self.namespaces: Dict[str, NamespaceConfig] = {}

    def parse_namespace(self, path: str) -> Dict[str, str]:
        """
        Parse namespace path into components.

        Args:
            path: Namespace path (e.g., "/team/engineering/api_design")

        Returns:
            Dict with type, scope, and name
        """
        parts = [p for p in path.split("/") if p]

        if not parts:
            return {"type": "global", "scope": None, "name": "default"}

        namespace_type = parts[0]

        if namespace_type == "global":
            return {
                "type": "global",
                "scope": None,
                "name": parts[1] if len(parts) > 1 else "default"
            }
        elif namespace_type in ["team", "user", "project"]:
            return {
                "type": namespace_type,
                "scope": parts[1] if len(parts) > 1 else None,
                "name": parts[2] if len(parts) > 2 else "default"
            }
        else:
            # Default to user namespace
            return {
                "type": "user",
                "scope": parts[0],
                "name": parts[1] if len(parts) > 1 else "default"
            }

    def build_namespace_path(
        self,
        namespace_type: str,
        name: str,
        scope: Optional[str] = None
    ) -> str:
        """
        Build namespace path from components.

        Args:
            namespace_type: Type (global, team, user, project)
            name: Name of the context
            scope: Scope ID (team_id, user_id, etc.)

        Returns:
            Full namespace path
        """
        if namespace_type == "global":
            return f"/global/{name}"
        elif namespace_type in ["team", "user", "project"]:
            if not scope:
                raise ValueError(f"{namespace_type} namespace requires scope")
            return f"/{namespace_type}/{scope}/{name}"
        else:
            raise ValueError(f"Unknown namespace type: {namespace_type}")

    def acquire_lock(
        self,
        namespace_path: str,
        lock_id: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Acquire a lock on a namespace.

        Args:
            namespace_path: Path to lock
            lock_id: Unique lock identifier
            ttl: Lock TTL in seconds

        Returns:
            True if lock acquired
        """
        if not self.redis_client:
            logger.warning("Redis not available, allowing lock")
            return True

        try:
            lock_key = f"namespace_lock:{namespace_path}"
            effective_ttl = ttl or self.lock_ttl

            # Try to set lock (NX = only if not exists)
            result = self.redis_client.set(
                lock_key,
                lock_id,
                nx=True,
                ex=effective_ttl
            )

            if result:
                logger.debug(f"Acquired lock on {namespace_path}")
                return True
            else:
                logger.debug(f"Lock already held on {namespace_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    def release_lock(self, namespace_path: str, lock_id: str) -> bool:
        """
        Release a lock on a namespace.

        Args:
            namespace_path: Path to unlock
            lock_id: Lock identifier (must match)

        Returns:
            True if lock released
        """
        if not self.redis_client:
            return True

        try:
            lock_key = f"namespace_lock:{namespace_path}"

            # Check if we own the lock
            current_lock = self.redis_client.get(lock_key)
            if current_lock and current_lock.decode('utf-8') == lock_id:
                self.redis_client.delete(lock_key)
                logger.debug(f"Released lock on {namespace_path}")
                return True
            else:
                logger.warning(f"Cannot release lock: not owner of {namespace_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False

    def is_locked(self, namespace_path: str) -> bool:
        """
        Check if namespace is locked.

        Args:
            namespace_path: Path to check

        Returns:
            True if locked
        """
        if not self.redis_client:
            return False

        try:
            lock_key = f"namespace_lock:{namespace_path}"
            return self.redis_client.exists(lock_key) > 0

        except Exception as e:
            logger.error(f"Failed to check lock: {e}")
            return False

    def get_namespace_contexts(
        self,
        namespace_path: str,
        neo4j_client=None
    ) -> List[Dict[str, Any]]:
        """
        Get all contexts in a namespace.

        Args:
            namespace_path: Namespace path
            neo4j_client: Neo4j client

        Returns:
            List of contexts
        """
        if not neo4j_client:
            return []

        try:
            query = """
            MATCH (c:Context)
            WHERE c.namespace = $namespace
            RETURN c
            ORDER BY c.created_at DESC
            """

            results = neo4j_client.query(query, {"namespace": namespace_path})
            return results if results else []

        except Exception as e:
            logger.error(f"Failed to get namespace contexts: {e}")
            return []

    def create_namespace(
        self,
        namespace_type: str,
        name: str,
        scope: Optional[str] = None,
        owner: Optional[str] = None,
        max_size: Optional[int] = None
    ) -> NamespaceConfig:
        """
        Create a new namespace configuration.

        Args:
            namespace_type: Type of namespace
            name: Namespace name
            scope: Scope ID
            owner: Owner user ID
            max_size: Maximum contexts allowed

        Returns:
            NamespaceConfig
        """
        path = self.build_namespace_path(namespace_type, name, scope)

        config = NamespaceConfig(
            name=name,
            path=path,
            default_ttl=self.NAMESPACE_TYPES[namespace_type]["default_ttl"],
            owner=owner,
            max_size=max_size,
            metadata={
                "type": namespace_type,
                "created_at": datetime.now().isoformat()
            }
        )

        self.namespaces[path] = config
        logger.info(f"Created namespace: {path}")

        return config

    def get_namespace_stats(self, neo4j_client=None) -> Dict[str, Any]:
        """
        Get statistics about namespaces.

        Args:
            neo4j_client: Neo4j client

        Returns:
            Statistics dictionary
        """
        if not neo4j_client:
            return {"error": "Neo4j not available"}

        try:
            query = """
            MATCH (c:Context)
            WHERE c.namespace IS NOT NULL
            WITH c.namespace as namespace, count(c) as context_count
            RETURN namespace, context_count
            ORDER BY context_count DESC
            """

            results = neo4j_client.query(query)

            stats = {
                "total_namespaces": len(results) if results else 0,
                "namespaces": {}
            }

            if results:
                for result in results:
                    namespace = result.get("namespace")
                    count = result.get("context_count", 0)
                    stats["namespaces"][namespace] = {
                        "context_count": count,
                        "type": self.parse_namespace(namespace)["type"]
                    }

            return stats

        except Exception as e:
            logger.error(f"Failed to get namespace stats: {e}")
            return {"error": str(e)}


def add_namespace_to_context(
    content: Dict[str, Any],
    namespace_path: Optional[str] = None,
    user_id: Optional[str] = None
) -> str:
    """
    Auto-assign namespace to context based on content or user.

    Args:
        content: Context content
        namespace_path: Explicit namespace path
        user_id: User ID for default namespace

    Returns:
        Namespace path
    """
    if namespace_path:
        return namespace_path

    # Auto-detect from content
    if "project_id" in content:
        return f"/project/{content['project_id']}/context"
    elif "team_id" in content:
        return f"/team/{content['team_id']}/context"
    elif user_id:
        return f"/user/{user_id}/context"
    else:
        return "/global/default"


__all__ = [
    "NamespaceManager",
    "NamespaceConfig",
    "add_namespace_to_context",
]
