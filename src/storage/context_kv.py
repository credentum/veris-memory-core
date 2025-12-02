#!/usr/bin/env python3
"""
context_kv.py: Context-aware key-value storage implementation

This module provides a context-aware wrapper around the base KV store
with additional features for context management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .kv_store import CacheEntry
from .kv_store import ContextKV as BaseContextKV
from .kv_store import MetricEvent


class ContextKV(BaseContextKV):
    """Enhanced context-aware KV store with additional context management features."""

    def __init__(
        self,
        config_path: str = ".ctxrc.yaml",
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ):
        """Initialize enhanced ContextKV."""
        super().__init__(config_path, verbose, config, test_mode)
        self.context_cache: Dict[str, Any] = {}

    def store_context(
        self, context_id: str, context_data: Dict[str, Any], ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store context data with optional TTL.

        Args:
            context_id: Unique identifier for the context
            context_data: Context data to store
            ttl_seconds: Optional TTL in seconds

        Returns:
            bool: True if stored successfully, False otherwise
        """
        if not context_id or not context_data:
            return False

        # Add metadata
        context_data["_stored_at"] = datetime.utcnow().isoformat()
        context_data["_context_id"] = context_id

        # Store in cache
        success = self.redis.set_cache(f"context:{context_id}", context_data, ttl_seconds)

        # Record metric
        if success:
            metric = MetricEvent(
                timestamp=datetime.utcnow(),
                metric_name="context.store",
                value=1.0,
                tags={"context_id": context_id},
                document_id=context_data.get("document_id"),
                agent_id=context_data.get("agent_id"),
            )
            self.redis.record_metric(metric)

        return success

    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context data by ID.

        Args:
            context_id: Context identifier

        Returns:
            Context data if found, None otherwise
        """
        if not context_id:
            return None

        # Try cache first
        cached = self.redis.get_cache(f"context:{context_id}")
        if cached:
            # Record hit metric
            metric = MetricEvent(
                timestamp=datetime.utcnow(),
                metric_name="context.get",
                value=1.0,
                tags={"context_id": context_id, "cache_hit": "true"},
                document_id=None,
                agent_id=None,
            )
            self.redis.record_metric(metric)
            return cached

        return None

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Redis-compatible set method for scratchpad operations.
        
        Args:
            key: Redis key
            value: Value to store
            ex: TTL in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        # ENHANCED DEBUG LOGGING
        print(f"🔍 ContextKV.set() called with key='{key}', value='{value[:50]}...', ex={ex}")
        
        try:
            # Check connection state
            print(f"🔍 Checking connection... ensure_connected()...")
            connection_ok = self.ensure_connected()
            print(f"🔍 ensure_connected() returned: {connection_ok}")
            
            print(f"🔍 Checking self.redis_client: {self.redis_client}")
            print(f"🔍 redis_client type: {type(self.redis_client)}")
            
            if not connection_ok or not self.redis_client:
                print("❌ Connection failed or redis_client is None")
                return False
                
            # Test Redis client methods exist
            print(f"🔍 Redis client has 'set' method: {hasattr(self.redis_client, 'set')}")
            
            # Use Redis client directly for compatibility (matches base class pattern)
            print(f"🔍 Calling redis_client.set('{key}', '{value}', ex={ex})...")
            if ex:
                result = self.redis_client.set(key, value, ex=ex)
            else:
                result = self.redis_client.set(key, value)
                
            print(f"🔍 Redis set result: {result} (type: {type(result)})")
            final_result = bool(result)
            print(f"🔍 Final return value: {final_result}")
            return final_result
            
        except Exception as e:
            print(f"❌ EXCEPTION in ContextKV.set(): {type(e).__name__}: {e}")
            import traceback
            print(f"❌ Full traceback:\n{traceback.format_exc()}")
            return False

    def get(self, key: str) -> Optional[str]:
        """Redis-compatible get method for scratchpad operations.
        
        Args:
            key: Redis key
            
        Returns:
            str: Retrieved value or None
        """
        try:
            # Ensure connection is established  
            if not self.ensure_connected() or not self.redis_client:
                return None
                
            result = self.redis_client.get(key)
            if result is None:
                return None
            if isinstance(result, bytes):
                return result.decode('utf-8')
            return str(result)
        except Exception as e:
            if self.verbose:
                print(f"KV get error: {e}")
            return None

    def exists(self, key: str) -> bool:
        """Redis-compatible exists method for scratchpad operations.
        
        Args:
            key: Redis key
            
        Returns:
            bool: True if key exists, False otherwise
        """
        try:
            # Ensure connection is established
            if not self.ensure_connected() or not self.redis_client:
                return False
                
            result = self.redis_client.exists(key)
            return bool(result)
        except Exception as e:
            if self.verbose:
                print(f"KV exists error: {e}")
            return False

    def connect(self, redis_password: Optional[str] = None) -> bool:
        """Connect to Redis with optional password.
        
        Args:
            redis_password: Optional Redis password
            
        Returns:
            bool: True if connected successfully
        """
        try:
            # Use base class connection establishment
            result = self.ensure_connected()
            if self.verbose and result:
                print("✅ ContextKV Redis connection established")
            elif self.verbose:
                print("⚠️ ContextKV Redis connection failed")
            return result and self.redis_client is not None
        except Exception as e:
            if self.verbose:
                print(f"KV connect error: {e}")
            return False

    def delete_context(self, context_id: str) -> bool:
        """Delete context data.

        Args:
            context_id: Context identifier

        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if not context_id:
            return False

        count = self.redis.delete_cache(f"context:{context_id}")

        if count > 0:
            # Record deletion metric
            metric = MetricEvent(
                timestamp=datetime.utcnow(),
                metric_name="context.delete",
                value=float(count),
                tags={"context_id": context_id},
                document_id=None,
                agent_id=None,
            )
            self.redis.record_metric(metric)

        return count > 0

    def list_contexts(self, pattern: str = "*") -> List[str]:
        """List all context IDs matching pattern.

        Args:
            pattern: Pattern to match context IDs

        Returns:
            List of matching context IDs
        """
        # This would need to be implemented in the base redis connector
        # For now, return empty list
        return []

    def update_context(self, context_id: str, updates: Dict[str, Any], merge: bool = True) -> bool:
        """Update existing context data.

        Args:
            context_id: Context identifier
            updates: Updates to apply
            merge: If True, merge with existing data; if False, replace

        Returns:
            bool: True if updated successfully, False otherwise
        """
        if not context_id or not updates:
            return False

        existing = self.get_context(context_id)
        if not existing:
            return False

        if merge:
            # Merge updates with existing data
            existing.update(updates)
            new_data = existing
        else:
            # Replace with updates
            new_data = updates

        # Preserve metadata
        new_data["_updated_at"] = datetime.utcnow().isoformat()
        new_data["_context_id"] = context_id

        return self.store_context(context_id, new_data)

    def get_context_metrics(self, context_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get metrics for a specific context.

        Args:
            context_id: Context identifier
            hours: Hours to look back

        Returns:
            Metrics summary for the context
        """
        end_time = datetime.utcnow()
        from datetime import timedelta

        start_time = end_time - timedelta(hours=hours)

        # Get metrics from Redis
        metrics = self.redis.get_metrics(f"context.{context_id}", start_time, end_time)

        return {
            "context_id": context_id,
            "period_hours": hours,
            "metrics_count": len(metrics),
            "metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "metric_name": m.metric_name,
                    "value": m.value,
                    "tags": m.tags,
                }
                for m in metrics
            ],
        }


# Export the enhanced class
__all__ = ["ContextKV", "MetricEvent", "CacheEntry"]
