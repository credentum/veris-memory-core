"""
Namespace utilities for Redis Bus - Application-Level Isolation.

Implements the same permission model as Context/Scratchpad:
- user_id from APIKeyInfo provides namespace isolation
- shared flag enables cross-team visibility
- include_shared flag controls retrieval filtering

Visibility Formula:
    VISIBLE = owned_by_current_user OR (is_shared AND wants_shared)
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NamespaceConfig:
    """Configuration for namespace-aware operations."""

    user_id: str
    include_shared: bool = True

    @property
    def private_prefix(self) -> str:
        """Get the user's private key prefix."""
        return f"{self.user_id}:"

    @property
    def shared_prefix(self) -> str:
        """Get the shared key prefix."""
        return "shared:"


def build_channel_key(pattern: str, user_id: str, **kwargs: str) -> str:
    """
    Build a namespaced channel key from pattern.

    Args:
        pattern: Channel pattern with {user_id} and other placeholders
        user_id: User ID from APIKeyInfo for namespace isolation
        **kwargs: Additional pattern parameters (plan_id, packet_id, etc.)

    Returns:
        Fully resolved channel key

    Raises:
        ValueError: If pattern has unresolved placeholders

    Example:
        >>> build_channel_key("{user_id}:work_packets:{plan_id}", "dev_team", plan_id="abc123")
        "dev_team:work_packets:abc123"
    """
    resolved = pattern.replace("{user_id}", user_id)
    for key, value in kwargs.items():
        resolved = resolved.replace(f"{{{key}}}", str(value))

    # Validate no unresolved placeholders (except wildcards)
    unresolved = re.findall(r"\{(\w+)\}", resolved)
    if unresolved:
        raise ValueError(f"Unresolved placeholders in channel key: {unresolved}")

    return resolved


def build_shared_channel_key(shared_pattern: str, **kwargs: str) -> str:
    """
    Build a shared channel key (no user_id prefix).

    Args:
        shared_pattern: Shared channel pattern (e.g., "shared:work_packets:{plan_id}")
        **kwargs: Pattern parameters

    Returns:
        Fully resolved shared channel key

    Raises:
        ValueError: If pattern has unresolved placeholders
    """
    resolved = shared_pattern
    for key, value in kwargs.items():
        resolved = resolved.replace(f"{{{key}}}", str(value))

    unresolved = re.findall(r"\{(\w+)\}", resolved)
    if unresolved:
        raise ValueError(f"Unresolved placeholders in shared channel key: {unresolved}")

    return resolved


def build_subscription_pattern(pattern: str, user_id: str, **kwargs: str) -> str:
    """
    Build a subscription pattern, replacing unspecified placeholders with wildcards.

    Args:
        pattern: Channel pattern with placeholders
        user_id: User ID from APIKeyInfo
        **kwargs: Known pattern parameters (unknown ones become "*")

    Returns:
        Pattern suitable for Redis PSUBSCRIBE

    Example:
        >>> build_subscription_pattern("{user_id}:work_packets:{plan_id}", "dev_team")
        "dev_team:work_packets:*"
    """
    resolved = pattern.replace("{user_id}", user_id)

    # Replace provided kwargs
    for key, value in kwargs.items():
        resolved = resolved.replace(f"{{{key}}}", str(value))

    # Replace remaining placeholders with wildcards
    resolved = re.sub(r"\{(\w+)\}", "*", resolved)

    return resolved


def get_subscription_patterns(
    channel_pattern: str,
    user_id: str,
    include_shared: bool = True,
    shared_pattern: Optional[str] = None,
    **kwargs: str,
) -> List[str]:
    """
    Get list of channel patterns to subscribe to based on namespace config.

    Args:
        channel_pattern: Private channel pattern
        user_id: User ID from APIKeyInfo
        include_shared: Whether to also subscribe to shared channel
        shared_pattern: Optional shared channel pattern
        **kwargs: Pattern parameters (unspecified become wildcards)

    Returns:
        List of patterns to subscribe to

    Example:
        >>> get_subscription_patterns(
        ...     "{user_id}:work_packets:{plan_id}",
        ...     "dev_team",
        ...     include_shared=True,
        ...     shared_pattern="shared:work_packets:{plan_id}"
        ... )
        ["dev_team:work_packets:*", "shared:work_packets:*"]
    """
    patterns = [build_subscription_pattern(channel_pattern, user_id, **kwargs)]

    if include_shared and shared_pattern:
        # Build shared pattern with wildcards for unspecified params
        shared_resolved = shared_pattern
        for key, value in kwargs.items():
            shared_resolved = shared_resolved.replace(f"{{{key}}}", str(value))
        shared_resolved = re.sub(r"\{(\w+)\}", "*", shared_resolved)
        patterns.append(shared_resolved)

    return patterns


def check_visibility(
    item_user_id: str,
    item_shared: bool,
    requester_user_id: str,
    include_shared: bool = True,
) -> bool:
    """
    Check if an item is visible to the requester.

    Implements the formula:
        VISIBLE = owned_by_current_user OR (is_shared AND wants_shared)

    This matches the existing Context/Scratchpad permission model.

    Args:
        item_user_id: The user_id that owns the item
        item_shared: Whether the item has shared=True
        requester_user_id: The user_id of the requester
        include_shared: Whether requester wants to see shared items

    Returns:
        True if item is visible to requester

    Examples:
        # Own data is always visible
        >>> check_visibility("team_a", False, "team_a", True)
        True

        # Other team's private data is never visible
        >>> check_visibility("team_a", False, "team_b", True)
        False

        # Other team's shared data is visible if include_shared=True
        >>> check_visibility("team_a", True, "team_b", True)
        True

        # Other team's shared data is NOT visible if include_shared=False
        >>> check_visibility("team_a", True, "team_b", False)
        False
    """
    owned_by_current_user = item_user_id == requester_user_id
    is_shared = item_shared
    wants_shared = include_shared

    return owned_by_current_user or (is_shared and wants_shared)


def extract_user_id_from_key(key: str) -> Optional[str]:
    """
    Extract user_id from a namespaced Redis key.

    Args:
        key: Redis key like "dev_team:work_packets:abc123"

    Returns:
        User ID or None if key doesn't match expected pattern

    Examples:
        >>> extract_user_id_from_key("dev_team:work_packets:abc123")
        "dev_team"

        >>> extract_user_id_from_key("shared:work_packets:abc123")
        None
    """
    if key.startswith("shared:"):
        return None  # Shared keys don't have user_id

    parts = key.split(":", 1)
    if len(parts) >= 1 and parts[0]:
        return parts[0]
    return None


def is_shared_key(key: str) -> bool:
    """
    Check if a Redis key is a shared channel key.

    Args:
        key: Redis key to check

    Returns:
        True if key starts with "shared:"
    """
    return key.startswith("shared:")


def validate_user_id(user_id: str) -> bool:
    """
    Validate that a user_id is safe for use in Redis keys.

    Args:
        user_id: User ID to validate

    Returns:
        True if user_id is valid

    Raises:
        ValueError: If user_id contains invalid characters
    """
    if not user_id:
        raise ValueError("user_id cannot be empty")

    # Allow alphanumeric, underscore, hyphen
    if not re.match(r"^[a-zA-Z0-9_-]+$", user_id):
        raise ValueError(
            f"user_id contains invalid characters: {user_id}. "
            "Only alphanumeric, underscore, and hyphen allowed."
        )

    if len(user_id) > 64:
        raise ValueError(f"user_id too long: {len(user_id)} chars (max 64)")

    return True
