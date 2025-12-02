#!/usr/bin/env python3
"""
cache_entry_validator.py: Cache entry validation module

This module provides validation for cache entries to ensure data integrity
and proper structure before storage and after retrieval.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


class CacheEntryValidator:
    """Validator for cache entries."""

    # Maximum sizes for various fields
    MAX_KEY_LENGTH = 256
    MAX_VALUE_SIZE = 1024 * 1024  # 1MB
    MAX_TTL_SECONDS = 86400 * 30  # 30 days
    MIN_TTL_SECONDS = 1

    # Key pattern (alphanumeric, dash, underscore, colon, dot)
    KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_\-:.]+$")

    @classmethod
    def validate_key(cls, key: str) -> tuple[bool, Optional[str]]:
        """Validate a cache key.

        Args:
            key: The cache key to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not key:
            return False, "Key cannot be empty"

        if len(key) > cls.MAX_KEY_LENGTH:
            return False, f"Key length exceeds maximum of {cls.MAX_KEY_LENGTH} characters"

        if not cls.KEY_PATTERN.match(key):
            return (
                False,
                "Key contains invalid characters (use alphanumeric, dash, underscore, colon, dot)",
            )

        return True, None

    @classmethod
    def validate_value(cls, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a cache value.

        Args:
            value: The value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            return False, "Value cannot be None"

        # Try to serialize to JSON to check size
        try:
            serialized = json.dumps(value, default=str)
            if len(serialized) > cls.MAX_VALUE_SIZE:
                return False, f"Value size exceeds maximum of {cls.MAX_VALUE_SIZE} bytes"
        except (TypeError, ValueError) as e:
            return False, f"Value cannot be serialized: {str(e)}"

        return True, None

    @classmethod
    def validate_ttl(cls, ttl_seconds: Optional[int]) -> tuple[bool, Optional[str]]:
        """Validate TTL (time-to-live) value.

        Args:
            ttl_seconds: TTL in seconds

        Returns:
            Tuple of (is_valid, error_message)
        """
        if ttl_seconds is None:
            # TTL is optional
            return True, None

        if not isinstance(ttl_seconds, int):
            return False, "TTL must be an integer"

        if ttl_seconds < cls.MIN_TTL_SECONDS:
            return False, f"TTL must be at least {cls.MIN_TTL_SECONDS} seconds"

        if ttl_seconds > cls.MAX_TTL_SECONDS:
            return False, f"TTL cannot exceed {cls.MAX_TTL_SECONDS} seconds"

        return True, None

    @classmethod
    def validate_cache_entry(cls, entry: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a complete cache entry.

        Args:
            entry: Cache entry dictionary with keys: key, value, ttl_seconds, etc.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate key
        if "key" not in entry:
            errors.append("Missing required field: key")
        else:
            is_valid, error = cls.validate_key(entry["key"])
            if not is_valid:
                errors.append(f"Invalid key: {error}")

        # Validate value
        if "value" not in entry:
            errors.append("Missing required field: value")
        else:
            is_valid, error = cls.validate_value(entry["value"])
            if not is_valid:
                errors.append(f"Invalid value: {error}")

        # Validate TTL if present
        if "ttl_seconds" in entry:
            is_valid, error = cls.validate_ttl(entry.get("ttl_seconds"))
            if not is_valid:
                errors.append(f"Invalid TTL: {error}")

        # Validate timestamps if present
        if "created_at" in entry:
            if not cls._validate_timestamp(entry["created_at"]):
                errors.append("Invalid created_at timestamp")

        if "last_accessed" in entry:
            if not cls._validate_timestamp(entry["last_accessed"]):
                errors.append("Invalid last_accessed timestamp")

        # Validate hit_count if present
        if "hit_count" in entry:
            if not isinstance(entry["hit_count"], int) or entry["hit_count"] < 0:
                errors.append("Invalid hit_count: must be a non-negative integer")

        return len(errors) == 0, errors

    @classmethod
    def _validate_timestamp(cls, timestamp: Any) -> bool:
        """Validate a timestamp value.

        Args:
            timestamp: Timestamp to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if isinstance(timestamp, datetime):
            return True

        if isinstance(timestamp, str):
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return True
            except (ValueError, AttributeError):
                return False

        return False

    @classmethod
    def sanitize_key(cls, key: str) -> str:
        """Sanitize a cache key by removing invalid characters.

        Args:
            key: The key to sanitize

        Returns:
            Sanitized key
        """
        # Replace invalid characters with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_\-:.]", "_", key)

        # Truncate if too long
        if len(sanitized) > cls.MAX_KEY_LENGTH:
            sanitized = sanitized[: cls.MAX_KEY_LENGTH]

        return sanitized

    @classmethod
    def create_valid_entry(
        cls, key: str, value: Any, ttl_seconds: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        """Create a validated cache entry.

        Args:
            key: Cache key
            value: Cache value
            ttl_seconds: Optional TTL
            **kwargs: Additional fields

        Returns:
            Validated cache entry dictionary

        Raises:
            ValueError: If validation fails
        """
        entry = {"key": key, "value": value, "created_at": datetime.utcnow(), "hit_count": 0}

        if ttl_seconds is not None:
            entry["ttl_seconds"] = ttl_seconds

        # Add any additional fields
        entry.update(kwargs)

        # Validate the entry
        is_valid, errors = cls.validate_cache_entry(entry)
        if not is_valid:
            raise ValueError(f"Invalid cache entry: {', '.join(errors)}")

        return entry


def validate_cache_key(key: str) -> bool:
    """Validate a cache key (convenience function).

    Args:
        key: The cache key to validate

    Returns:
        bool: True if valid, False otherwise
    """
    is_valid, _ = CacheEntryValidator.validate_key(key)
    return is_valid


def validate_cache_value(value: Any) -> bool:
    """Validate a cache value (convenience function).

    Args:
        value: The value to validate

    Returns:
        bool: True if valid, False otherwise
    """
    is_valid, _ = CacheEntryValidator.validate_value(value)
    return is_valid


def validate_ttl(ttl_seconds: Optional[int]) -> bool:
    """Validate a TTL value (convenience function).

    Args:
        ttl_seconds: TTL in seconds

    Returns:
        bool: True if valid, False otherwise
    """
    is_valid, _ = CacheEntryValidator.validate_ttl(ttl_seconds)
    return is_valid


# Export classes and functions
__all__ = ["CacheEntryValidator", "validate_cache_key", "validate_cache_value", "validate_ttl"]
