#!/usr/bin/env python3
"""
kv_validators.py: Input validation for KV store operations
"""

from datetime import datetime
from typing import Any, Dict


def validate_cache_entry(data: Dict[str, Any]) -> bool:
    """Validate cache entry structure"""
    if not isinstance(data, dict):
        return False

    required_fields = ["key", "value", "created_at", "ttl_seconds"]

    for field in required_fields:
        if field not in data:
            return False

    # Validate types
    if not isinstance(data.get("key"), str):
        return False

    if not isinstance(data.get("ttl_seconds"), int) or data["ttl_seconds"] < 0:
        return False

    # Validate timestamp
    try:
        datetime.fromisoformat(data["created_at"])
    except (ValueError, TypeError):
        return False

    return True


def validate_metric_event(metric: Dict[str, Any]) -> bool:
    """Validate metric event structure"""
    if not isinstance(metric, dict):
        return False

    required_fields = ["timestamp", "metric_name", "value", "tags"]

    for field in required_fields:
        if field not in metric:
            return False

    # Validate types
    if not isinstance(metric.get("metric_name"), str):
        return False

    if not isinstance(metric.get("value"), (int, float)):
        return False

    if not isinstance(metric.get("tags"), dict):
        return False

    # Validate timestamp
    if isinstance(metric["timestamp"], str):
        try:
            datetime.fromisoformat(metric["timestamp"])
        except ValueError:
            return False
    elif not isinstance(metric["timestamp"], datetime):
        return False

    return True


def sanitize_metric_name(name: str) -> str:
    """Sanitize metric name to prevent injection"""
    # Allow only alphanumeric, dots, underscores, and hyphens
    import re

    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name)

    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]

    return sanitized


def validate_time_range(start_time: datetime, end_time: datetime, max_days: int = 90) -> bool:
    """Validate time range for queries"""
    if start_time >= end_time:
        return False

    # Check maximum range
    delta = end_time - start_time
    if delta.days > max_days:
        return False

    # Don't allow future dates
    now = datetime.utcnow()
    if end_time > now:
        return False

    return True


def validate_redis_key(key: str) -> bool:
    """Validate Redis key format"""
    if not key or not isinstance(key, str):
        return False

    # Check length (Redis max is 512MB but we'll be reasonable)
    if len(key) > 1024:
        return False

    # Check for control characters
    if any(ord(c) < 32 for c in key):
        return False

    return True


def validate_session_data(data: Any) -> bool:
    """Validate session data structure"""
    if not isinstance(data, dict):
        return False

    # Limit session data size (e.g., 1MB)
    import json

    try:
        serialized = json.dumps(data)
        if len(serialized) > 1024 * 1024:  # 1MB
            return False
    except (TypeError, ValueError):
        return False

    return True
