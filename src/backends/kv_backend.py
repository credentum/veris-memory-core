#!/usr/bin/env python3
"""
Key-Value backend adapter for Redis-based storage.

This module implements the BackendSearchInterface for Redis key-value operations,
providing fast lookup capabilities for agent state and session data.
"""

import json
import time
from typing import List, Dict, Any, Optional

from ..interfaces.backend_interface import BackendSearchInterface, SearchOptions, BackendHealthStatus, BackendSearchError
from ..interfaces.memory_result import MemoryResult, ResultSource, ContentType
from ..utils.logging_middleware import backend_logger, log_backend_timing


class KVBackend(BackendSearchInterface):
    """
    Key-Value search backend implementation using Redis.
    
    Provides fast lookup capabilities for agent state, session data,
    and transient information with TTL support.
    """
    
    def __init__(self, kv_store=None, simple_redis=None):
        """
        Initialize KV backend.
        
        Args:
            kv_store: ContextKV instance for structured operations
            simple_redis: SimpleRedisClient for direct Redis access
        """
        self.kv_store = kv_store
        self.simple_redis = simple_redis or (kv_store.redis if kv_store else None)
        
        # Key prefixes for different data types
        self.prefixes = {
            "agent_state": "state:",
            "scratchpad": "scratchpad:",
            "session": "session:",
            "cache": "cache:",
            "fact": "fact:"
        }
    
    @property
    def backend_name(self) -> str:
        """Return the name of this backend."""
        return "kv"
    
    async def search(self, query: str, options: SearchOptions) -> List[MemoryResult]:
        """
        Search key-value store using pattern matching and key scanning.
        
        Args:
            query: Search query (treated as key pattern)
            options: Search configuration options
            
        Returns:
            List of MemoryResult objects from matching keys
            
        Raises:
            BackendSearchError: If KV search operation fails
        """
        async with log_backend_timing(self.backend_name, "search", backend_logger) as metadata:
            try:
                # Determine search strategy based on query and options
                search_results = []
                
                # Strategy 1: Direct key lookup if query looks like a key
                if self._looks_like_key(query):
                    direct_result = await self._direct_key_lookup(query, options)
                    if direct_result:
                        search_results.extend(direct_result)
                
                # Strategy 2: Namespace-based search
                if options.namespace:
                    namespace_results = await self._namespace_search(options.namespace, query, options)
                    search_results.extend(namespace_results)
                
                # Strategy 3: Pattern-based key scanning
                pattern_results = await self._pattern_search(query, options)
                search_results.extend(pattern_results)
                
                # Strategy 4: Content-based search in structured values
                if len(search_results) < options.limit:
                    content_results = await self._content_search(query, options)
                    search_results.extend(content_results)
                
                # Deduplicate and sort results
                unique_results = self._deduplicate_results(search_results)
                filtered_results = self._apply_filters(unique_results, options)
                
                metadata["search_strategies_used"] = 4
                metadata["total_found"] = len(search_results)
                metadata["after_deduplication"] = len(unique_results)
                metadata["result_count"] = len(filtered_results)
                metadata["top_score"] = filtered_results[0].score if filtered_results else 0.0
                
                backend_logger.info(
                    f"KV search completed",
                    query_type="multi_strategy",
                    **metadata
                )
                
                return filtered_results
                
            except Exception as e:
                error_msg = f"KV search failed: {str(e)}"
                backend_logger.error(error_msg, error=str(e))
                raise BackendSearchError(self.backend_name, error_msg, e)
    
    async def health_check(self) -> BackendHealthStatus:
        """
        Check the health of the KV backend.
        
        Returns:
            BackendHealthStatus with current health information
        """
        start_time = time.time()
        
        try:
            # Test Redis connectivity
            if self.simple_redis and hasattr(self.simple_redis, 'redis_client'):
                redis_client = self.simple_redis.redis_client
                ping_result = redis_client.ping()
                connectivity_ok = ping_result is True
            elif self.simple_redis:
                # Try basic operation
                test_key = f"health_check_{int(time.time())}"
                self.simple_redis.set(test_key, "test", ex=5)  # 5 second TTL
                test_value = self.simple_redis.get(test_key)
                connectivity_ok = test_value == "test"
                self.simple_redis.delete(test_key)  # Cleanup
            else:
                connectivity_ok = False
            
            response_time = (time.time() - start_time) * 1000
            
            # Get basic Redis info
            redis_info = {}
            if connectivity_ok and self.simple_redis and hasattr(self.simple_redis, 'redis_client'):
                try:
                    info = self.simple_redis.redis_client.info()
                    redis_info = {
                        "version": info.get("redis_version", "unknown"),
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory_human": info.get("used_memory_human", "unknown"),
                        "keyspace_hits": info.get("keyspace_hits", 0),
                        "keyspace_misses": info.get("keyspace_misses", 0)
                    }
                except Exception as e:
                    backend_logger.warning(f"Failed to get Redis info: {e}")
            
            # Count keys by prefix
            key_counts = {}
            if connectivity_ok:
                try:
                    for prefix_name, prefix in self.prefixes.items():
                        pattern = f"{prefix}*"
                        count = len(self._scan_keys(pattern, limit=100))  # Sample count
                        key_counts[prefix_name] = count
                except Exception as e:
                    backend_logger.warning(f"Failed to count keys: {e}")
            
            status = "healthy" if connectivity_ok else "unhealthy"
            
            return BackendHealthStatus(
                status=status,
                response_time_ms=response_time,
                metadata={
                    "connectivity_test_passed": connectivity_ok,
                    "redis_info": redis_info,
                    "key_counts": key_counts,
                    "prefixes_configured": list(self.prefixes.keys()),
                    "kv_store_available": self.kv_store is not None,
                    "simple_redis_available": self.simple_redis is not None
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return BackendHealthStatus(
                status="unhealthy",
                response_time_ms=response_time,
                error_message=str(e),
                metadata={
                    "connection_error": True,
                    "error_type": type(e).__name__
                }
            )
    
    async def initialize(self) -> None:
        """Initialize the KV backend."""
        backend_logger.info("Initializing KV backend")
        
        try:
            # Test connectivity
            health = await self.health_check()
            if health.status == "unhealthy":
                raise Exception(f"Health check failed: {health.error_message}")
            
            backend_logger.info("KV backend initialized successfully")
            
        except Exception as e:
            backend_logger.error(f"KV backend initialization failed: {e}")
            raise BackendSearchError(self.backend_name, f"Initialization failed: {e}", e)
    
    async def cleanup(self) -> None:
        """Clean up KV backend resources."""
        backend_logger.info("Cleaning up KV backend")
        # Redis client cleanup handled by the client itself
    
    # Public method for direct key operations
    async def get_by_key(self, key: str, namespace: Optional[str] = None) -> Optional[MemoryResult]:
        """
        Get a specific key's value as a MemoryResult.
        
        Args:
            key: The Redis key to retrieve
            namespace: Optional namespace prefix
            
        Returns:
            MemoryResult if key exists, None otherwise
        """
        try:
            full_key = f"{namespace}:{key}" if namespace else key
            
            if self.simple_redis:
                value = self.simple_redis.get(full_key)
                if value:
                    return self._convert_value_to_result(full_key, value)
            
            return None
            
        except Exception as e:
            backend_logger.error(f"Direct key lookup failed: {e}", key=key)
            raise BackendSearchError(self.backend_name, f"Key lookup failed: {e}", e)
    
    async def set_key(self, key: str, content: str, ttl: Optional[int] = None, namespace: Optional[str] = None) -> bool:
        """
        Set a key's value with optional TTL.
        
        Args:
            key: The Redis key to set
            content: Content to store
            ttl: Optional time-to-live in seconds
            namespace: Optional namespace prefix
            
        Returns:
            True if successful
        """
        try:
            full_key = f"{namespace}:{key}" if namespace else key
            
            if self.simple_redis:
                if ttl:
                    return self.simple_redis.set(full_key, content, ex=ttl)
                else:
                    return self.simple_redis.set(full_key, content)
            
            return False
            
        except Exception as e:
            backend_logger.error(f"Key set operation failed: {e}", key=key)
            raise BackendSearchError(self.backend_name, f"Key set failed: {e}", e)
    
    # Private helper methods
    
    def _looks_like_key(self, query: str) -> bool:
        """Check if query looks like a direct key lookup."""
        # Keys typically have colons or are structured
        return ":" in query or any(query.startswith(prefix) for prefix in self.prefixes.values())
    
    async def _direct_key_lookup(self, key: str, options: SearchOptions) -> List[MemoryResult]:
        """Perform direct key lookup."""
        try:
            if not self.simple_redis:
                return []
            
            value = self.simple_redis.get(key)
            if value:
                result = self._convert_value_to_result(key, value)
                if result:
                    return [result]
            
            return []
            
        except Exception as e:
            backend_logger.warning(f"Direct key lookup failed: {e}")
            return []
    
    async def _namespace_search(self, namespace: str, query: str, options: SearchOptions) -> List[MemoryResult]:
        """Search within a specific namespace."""
        try:
            pattern = f"{namespace}:*"
            matching_keys = self._scan_keys(pattern, limit=options.limit * 2)
            
            results = []
            for key in matching_keys:
                try:
                    value = self.simple_redis.get(key) if self.simple_redis else None
                    if value and query.lower() in value.lower():
                        result = self._convert_value_to_result(key, value)
                        if result:
                            results.append(result)
                except Exception as e:
                    backend_logger.warning(f"Failed to process key {key}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            backend_logger.warning(f"Namespace search failed: {e}")
            return []
    
    async def _pattern_search(self, query: str, options: SearchOptions) -> List[MemoryResult]:
        """Search using Redis key patterns."""
        try:
            # Try different pattern strategies
            patterns = [
                f"*{query}*",  # Contains query
                f"{query}*",  # Starts with query
                f"*:{query}*"  # Contains in value part
            ]
            
            all_results = []
            for pattern in patterns:
                matching_keys = self._scan_keys(pattern, limit=options.limit)
                for key in matching_keys:
                    try:
                        value = self.simple_redis.get(key) if self.simple_redis else None
                        if value:
                            result = self._convert_value_to_result(key, value)
                            if result:
                                all_results.append(result)
                    except Exception as e:
                        backend_logger.warning(f"Failed to process pattern match {key}: {e}")
                        continue
                
                if len(all_results) >= options.limit:
                    break
            
            return all_results
            
        except Exception as e:
            backend_logger.warning(f"Pattern search failed: {e}")
            return []
    
    async def _content_search(self, query: str, options: SearchOptions) -> List[MemoryResult]:
        """Search within stored content values."""
        try:
            # Sample keys from different prefixes
            sample_keys = []
            for prefix in self.prefixes.values():
                pattern_keys = self._scan_keys(f"{prefix}*", limit=50)
                sample_keys.extend(pattern_keys)
            
            results = []
            for key in sample_keys[:min(200, len(sample_keys))]:  # Limit scanning
                try:
                    value = self.simple_redis.get(key) if self.simple_redis else None
                    if value and query.lower() in value.lower():
                        result = self._convert_value_to_result(key, value)
                        if result:
                            # Boost score based on content relevance
                            query_matches = value.lower().count(query.lower())
                            result.score = min(1.0, 0.5 + (query_matches * 0.1))
                            results.append(result)
                except Exception as e:
                    backend_logger.warning(f"Failed to search content in {key}: {e}")
                    continue
                
                if len(results) >= options.limit:
                    break
            
            return results
            
        except Exception as e:
            backend_logger.warning(f"Content search failed: {e}")
            return []
    
    def _scan_keys(self, pattern: str, limit: int = 100) -> List[str]:
        """Scan Redis keys matching pattern."""
        try:
            if not self.simple_redis or not hasattr(self.simple_redis, 'redis_client'):
                return []
            
            redis_client = self.simple_redis.redis_client
            cursor = 0
            keys = []
            
            while len(keys) < limit:
                cursor, batch_keys = redis_client.scan(cursor=cursor, match=pattern, count=50)
                keys.extend([key.decode() if isinstance(key, bytes) else key for key in batch_keys])
                
                if cursor == 0:  # Full scan completed
                    break
            
            return keys[:limit]
            
        except Exception as e:
            backend_logger.warning(f"Key scanning failed: {e}")
            return []
    
    def _convert_value_to_result(self, key: str, value: Any) -> Optional[MemoryResult]:
        """Convert Redis key-value pair to MemoryResult."""
        try:
            # Determine content type based on key prefix
            content_type = ContentType.GENERAL
            for prefix_name, prefix in self.prefixes.items():
                if key.startswith(prefix):
                    if prefix_name == "agent_state":
                        content_type = ContentType.PERSONAL_INFO
                    elif prefix_name == "fact":
                        content_type = ContentType.FACT
                    break
            
            # Parse value if it's JSON
            parsed_value = value
            metadata = {}
            
            try:
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, dict):
                        metadata = parsed_value
            except json.JSONDecodeError:
                # Not JSON, treat as plain text
                pass
            
            # Extract text content
            if isinstance(parsed_value, dict):
                text = parsed_value.get('text', parsed_value.get('content', str(parsed_value)))
            else:
                text = str(parsed_value)
            
            # Extract namespace from key
            namespace = None
            if ':' in key:
                parts = key.split(':', 1)
                if len(parts) == 2 and parts[0] in ['state', 'scratchpad', 'session', 'cache', 'fact']:
                    namespace = parts[0]
            
            # Calculate relevance score
            score = 0.8  # KV results have good relevance since they're direct matches
            
            # Create result
            result = MemoryResult(
                id=key,
                text=text,
                type=content_type,
                score=score,
                source=ResultSource.KV,
                tags=[],
                metadata={
                    **metadata,
                    'redis_key': key,
                    'kv_search': True,
                    'value_type': type(parsed_value).__name__
                },
                namespace=namespace,
                title=key.split(':')[-1] if ':' in key else key  # Use key suffix as title
            )
            
            return result
            
        except Exception as e:
            backend_logger.warning(f"Failed to convert KV result: {e}", key=key)
            return None
    
    def _deduplicate_results(self, results: List[MemoryResult]) -> List[MemoryResult]:
        """Remove duplicate results based on ID."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        return unique_results
    
    def _apply_filters(self, results: List[MemoryResult], options: SearchOptions) -> List[MemoryResult]:
        """Apply filtering and sorting to results."""
        filtered = results
        
        # Apply score threshold
        if options.score_threshold > 0:
            filtered = [r for r in filtered if r.score >= options.score_threshold]
        
        # Apply type filter
        if options.filters.get("type"):
            filtered = [r for r in filtered if r.type == options.filters["type"]]
        
        # Sort by score (highest first) then by key name for deterministic ordering
        filtered.sort(key=lambda x: (-x.score, x.id))
        
        return filtered[:options.limit]