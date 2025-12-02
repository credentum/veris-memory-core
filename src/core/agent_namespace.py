"""
Agent Namespace Management for MCP Server.

This module provides namespace isolation for agent data, ensuring that
each agent can only access its own data and preventing cross-agent
data leakage.
"""

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class NamespaceError(Exception):
    """Raised when namespace operations fail."""

    def __init__(self, message: str, error_type: str = "namespace_error"):
        super().__init__(message)
        self.error_type = error_type


@dataclass
class AgentSession:
    """Represents an active agent session."""

    agent_id: str
    session_id: str
    created_at: datetime
    last_accessed: datetime
    metadata: Dict[str, Any]

    @property
    def is_expired(self) -> bool:
        """Check if session has expired (24 hours)."""
        return datetime.utcnow() - self.last_accessed > timedelta(hours=24)

    def update_access_time(self):
        """Update the last accessed timestamp."""
        self.last_accessed = datetime.utcnow()


class AgentNamespace:
    """
    Manages agent namespace isolation for data access.

    This class provides secure namespace management to ensure agents
    can only access their own data. It implements the namespace pattern:

    - Scratchpad: agent:{agent_id}:scratchpad:{key}
    - State: agent:{agent_id}:state:{key}
    - Session: agent:{agent_id}:session:{session_id}
    - Memory: agent:{agent_id}:memory:{memory_type}:{key}
    """

    # Valid namespace prefixes
    VALID_PREFIXES = {"scratchpad", "state", "session", "memory", "config", "temp"}

    # Agent ID validation pattern
    AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

    # Key validation pattern (no special characters that could cause issues)
    KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")

    def __init__(self):
        """Initialize the namespace manager."""
        self._active_sessions: Dict[str, AgentSession] = {}
        self._agent_permissions: Dict[str, Set[str]] = {}

    def validate_agent_id(self, agent_id: str) -> bool:
        """
        Validate agent ID format.

        Args:
            agent_id: Agent identifier to validate

        Returns:
            True if valid, False otherwise
        """
        if not agent_id or not isinstance(agent_id, str):
            return False

        return bool(self.AGENT_ID_PATTERN.match(agent_id))

    def validate_key(self, key: str) -> bool:
        """
        Validate namespace key format.

        Args:
            key: Key to validate

        Returns:
            True if valid, False otherwise
        """
        if not key or not isinstance(key, str):
            return False

        return bool(self.KEY_PATTERN.match(key))

    def validate_prefix(self, prefix: str) -> bool:
        """
        Validate namespace prefix.

        Args:
            prefix: Prefix to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(prefix, str):
            return False
        return prefix in self.VALID_PREFIXES

    def create_namespaced_key(self, agent_id: str, prefix: str, key: str) -> str:
        """
        Create a namespaced key for an agent.

        Args:
            agent_id: Agent identifier
            prefix: Namespace prefix (scratchpad, state, etc.)
            key: Resource key

        Returns:
            Fully namespaced key

        Raises:
            NamespaceError: If validation fails
        """
        # Validate inputs
        if not self.validate_agent_id(agent_id):
            raise NamespaceError(f"Invalid agent ID: {agent_id}", error_type="invalid_agent_id")

        if not self.validate_prefix(prefix):
            raise NamespaceError(f"Invalid namespace prefix: {prefix}", error_type="invalid_prefix")

        if not self.validate_key(key):
            raise NamespaceError(f"Invalid key format: {key}", error_type="invalid_key")

        # Create the namespaced key
        namespaced_key = f"agent:{agent_id}:{prefix}:{key}"

        logger.debug(f"Created namespaced key: {namespaced_key}")
        return namespaced_key

    def parse_namespaced_key(self, namespaced_key: str) -> tuple:
        """
        Parse a namespaced key back into components.

        Args:
            namespaced_key: The namespaced key to parse

        Returns:
            Tuple of (agent_id, prefix, key)

        Raises:
            NamespaceError: If key format is invalid
        """
        parts = namespaced_key.split(":")

        if len(parts) != 4 or parts[0] != "agent":
            raise NamespaceError(
                f"Invalid namespaced key format: {namespaced_key}",
                error_type="invalid_key_format",
            )

        _, agent_id, prefix, key = parts

        # Validate parsed components
        if not self.validate_agent_id(agent_id):
            raise NamespaceError(
                f"Invalid agent ID in key: {agent_id}", error_type="invalid_agent_id"
            )

        if not self.validate_prefix(prefix):
            raise NamespaceError(f"Invalid prefix in key: {prefix}", error_type="invalid_prefix")

        return agent_id, prefix, key

    def verify_agent_access(self, agent_id: str, namespaced_key: str) -> bool:
        """
        Verify that an agent has access to a specific namespaced key.

        Args:
            agent_id: Agent requesting access
            namespaced_key: Key being accessed

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            key_agent_id, prefix, key = self.parse_namespaced_key(namespaced_key)

            # Agent can only access their own namespace
            if key_agent_id != agent_id:
                logger.warning(f"Agent {agent_id} attempted to access {key_agent_id}'s data")
                return False

            # Check if agent has permission for this prefix
            agent_permissions = self._agent_permissions.get(agent_id, set())
            if agent_permissions and prefix not in agent_permissions:
                logger.warning(f"Agent {agent_id} lacks permission for prefix: {prefix}")
                return False

            return True

        except NamespaceError as e:
            logger.error(f"Namespace validation error: {e}")
            return False

    def create_agent_session(self, agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """
        Create a new session for an agent.

        Args:
            agent_id: Agent identifier
            metadata: Optional session metadata

        Returns:
            Session ID

        Raises:
            NamespaceError: If agent ID is invalid
        """
        if not self.validate_agent_id(agent_id):
            raise NamespaceError(f"Invalid agent ID: {agent_id}", error_type="invalid_agent_id")

        session_id = str(uuid.uuid4())
        session = AgentSession(
            agent_id=agent_id,
            session_id=session_id,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            metadata=metadata or {},
        )

        session_key = self.create_namespaced_key(agent_id, "session", session_id)
        self._active_sessions[session_key] = session

        logger.info(f"Created session {session_id} for agent {agent_id}")
        return session_id

    def get_agent_session(self, agent_id: str, session_id: str) -> Optional[AgentSession]:
        """
        Get an agent session.

        Args:
            agent_id: Agent identifier
            session_id: Session identifier

        Returns:
            AgentSession if found and valid, None otherwise
        """
        try:
            session_key = self.create_namespaced_key(agent_id, "session", session_id)
            session = self._active_sessions.get(session_key)

            if session and not session.is_expired:
                session.update_access_time()
                return session
            elif session and session.is_expired:
                # Clean up expired session
                del self._active_sessions[session_key]
                logger.info(f"Cleaned up expired session {session_id} for agent {agent_id}")

            return None

        except NamespaceError:
            return None

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        expired_keys = []

        for key, session in self._active_sessions.items():
            if session.is_expired:
                expired_keys.append(key)

        for key in expired_keys:
            del self._active_sessions[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired sessions")

        return len(expired_keys)

    def set_agent_permissions(self, agent_id: str, permissions: Set[str]) -> None:
        """
        Set namespace permissions for an agent.

        Args:
            agent_id: Agent identifier
            permissions: Set of allowed namespace prefixes
        """
        if not self.validate_agent_id(agent_id):
            raise NamespaceError(f"Invalid agent ID: {agent_id}", error_type="invalid_agent_id")

        # Validate all permissions
        invalid_permissions = permissions - self.VALID_PREFIXES
        if invalid_permissions:
            raise NamespaceError(
                f"Invalid permissions: {invalid_permissions}",
                error_type="invalid_permissions",
            )

        self._agent_permissions[agent_id] = permissions.copy()
        logger.info(f"Set permissions for agent {agent_id}: {permissions}")

    def get_agent_permissions(self, agent_id: str) -> Set[str]:
        """
        Get namespace permissions for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Set of allowed namespace prefixes
        """
        return self._agent_permissions.get(agent_id, self.VALID_PREFIXES.copy())

    def list_agent_keys(self, agent_id: str, prefix: str = None) -> List[str]:
        """
        List all keys for an agent (useful for debugging/admin).

        Args:
            agent_id: Agent identifier
            prefix: Optional prefix filter

        Returns:
            List of namespaced keys for the agent
        """
        if not self.validate_agent_id(agent_id):
            raise NamespaceError(f"Invalid agent ID: {agent_id}", error_type="invalid_agent_id")

        agent_prefix = f"agent:{agent_id}:"

        if prefix:
            if not self.validate_prefix(prefix):
                raise NamespaceError(f"Invalid prefix: {prefix}", error_type="invalid_prefix")
            agent_prefix += f"{prefix}:"

        # In a real implementation, this would query the storage backend
        # For now, return the pattern that would be used
        return [f"{agent_prefix}*"]

    def get_namespace_stats(self) -> Dict[str, Any]:
        """
        Get statistics about namespace usage.

        Returns:
            Dictionary with namespace statistics
        """
        active_agents = set()
        prefix_counts: Dict[str, int] = {}

        for key in self._active_sessions.keys():
            try:
                agent_id, prefix, _ = self.parse_namespaced_key(key)
                active_agents.add(agent_id)
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            except NamespaceError:
                continue

        return {
            "active_agents": len(active_agents),
            "active_sessions": len(self._active_sessions),
            "prefix_usage": prefix_counts,
            "configured_permissions": len(self._agent_permissions),
        }
