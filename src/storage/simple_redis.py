#!/usr/bin/env python3
"""
simple_redis.py: Direct Redis client for scratchpad operations

This module provides a simple, direct Redis interface that bypasses
the complex ContextKV inheritance chain to ensure reliable scratchpad operations.
"""

import os
from typing import Optional
import redis
import logging

logger = logging.getLogger(__name__)


class SimpleRedisClient:
    """Simple, direct Redis client for reliable scratchpad operations."""
    
    def __init__(self):
        """Initialize Redis client with environment configuration."""
        self.client: Optional[redis.Redis] = None
        self.is_connected = False
        
    def connect(self, redis_password: Optional[str] = None) -> bool:
        """
        Connect to Redis with minimal complexity.
        
        Args:
            redis_password: Optional Redis password
            
        Returns:
            bool: True if connected successfully
        """
        try:
            # Get Redis configuration from environment
            # First check REDIS_URL for Docker deployments
            redis_url = os.getenv("REDIS_URL")
            password_from_url = None
            if redis_url:
                # Parse redis://:password@host:port/db format
                import re
                url_match = re.match(r"^redis://(?::([^@]+)@)?([^:/]+):?(\d+)?/?(\d+)?", redis_url)
                if url_match:
                    password_from_url = url_match.group(1)  # Password (if present)
                    host = url_match.group(2)
                    port = int(url_match.group(3)) if url_match.group(3) else 6379
                    db = int(url_match.group(4)) if url_match.group(4) else 0
                else:
                    # Fallback to individual env vars
                    logger.warning(f"Failed to parse REDIS_URL: {redis_url}, falling back to individual env vars")
                    host = os.getenv("REDIS_HOST", "redis")
                    port = int(os.getenv("REDIS_PORT", "6379"))
                    db = int(os.getenv("REDIS_DB", "0"))
            else:
                # Use individual environment variables
                host = os.getenv("REDIS_HOST", "redis")
                port = int(os.getenv("REDIS_PORT", "6379"))
                db = int(os.getenv("REDIS_DB", "0"))
            
            # Create connection parameters
            connection_params = {
                "host": host,
                "port": port,
                "db": db,
                "decode_responses": True,  # Return strings instead of bytes
                "socket_connect_timeout": 5,
                "socket_timeout": 5,
                "retry_on_timeout": True
            }
            
            # Add password with priority: parameter > URL > env var
            password = redis_password or password_from_url or os.getenv("REDIS_PASSWORD")
            if password:
                connection_params["password"] = password
                logger.debug("Using Redis password authentication")
            
            # Create Redis client
            self.client = redis.Redis(**connection_params)
            
            # Test connection
            self.client.ping()
            self.is_connected = True

            logger.info(f"✅ SimpleRedisClient connected to {host}:{port}")
            logger.debug(f"Connected to Redis at {host}:{port} db={db}")
            return True
            
        except Exception as e:
            logger.error(f"❌ SimpleRedisClient connection failed: {e}")
            self.client = None
            self.is_connected = False
            return False
    
    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """
        Set a key-value pair in Redis with optional TTL.
        
        Args:
            key: Redis key
            value: Value to store
            ex: TTL in seconds (optional)
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.is_connected or not self.client:
                logger.warning("Redis not connected, attempting reconnection...")
                if not self.connect():
                    return False
            
            # Perform Redis SET operation
            if ex:
                result = self.client.set(key, value, ex=ex)
            else:
                result = self.client.set(key, value)
            
            return bool(result)
            
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in set(): {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error in set(): {e}")
            return False

    def setex(self, key: str, time: int, value: str) -> bool:
        """
        Set a key-value pair in Redis with TTL (expiration time).

        This method uses Redis SETEX command for atomic set-with-expiration.
        Note: argument order matches Redis python client (key, time, value).

        Args:
            key: Redis key
            time: TTL in seconds
            value: Value to store

        Returns:
            bool: True if successful
        """
        try:
            if not self.is_connected or not self.client:
                logger.warning("Redis not connected, attempting reconnection...")
                if not self.connect():
                    return False

            # Perform Redis SETEX operation
            result = self.client.setex(key, time, value)
            return bool(result)

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in setex(): {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error in setex(): {e}")
            return False

    def get(self, key: str) -> Optional[str]:
        """
        Get a value from Redis by key.
        
        Args:
            key: Redis key
            
        Returns:
            str: Retrieved value or None
        """
        try:
            if not self.is_connected or not self.client:
                logger.warning("Redis not connected, attempting reconnection...")
                if not self.connect():
                    return None
            
            # Perform Redis GET operation
            result = self.client.get(key)
            return result  # Already decoded to string due to decode_responses=True
            
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in get(): {e}")
            self.is_connected = False
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get(): {e}")
            return None
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Redis key
            
        Returns:
            bool: True if key exists
        """
        try:
            if not self.is_connected or not self.client:
                logger.warning("Redis not connected, attempting reconnection...")
                if not self.connect():
                    return False
            
            # Perform Redis EXISTS operation
            result = self.client.exists(key)
            return bool(result)
            
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in exists(): {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error in exists(): {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Redis key
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            if not self.is_connected or not self.client:
                logger.warning("Redis not connected, attempting reconnection...")
                if not self.connect():
                    return False
            
            # Perform Redis DELETE operation
            result = self.client.delete(key)
            return bool(result)
            
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in delete(): {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error in delete(): {e}")
            return False
    
    def keys(self, pattern: str = "*") -> list:
        """
        Get keys matching a pattern.
        
        Args:
            pattern: Redis key pattern (default: "*")
            
        Returns:
            list: List of matching keys
        """
        try:
            if not self.is_connected or not self.client:
                logger.warning("Redis not connected, attempting reconnection...")
                if not self.connect():
                    return []
            
            # Perform Redis KEYS operation
            result = self.client.keys(pattern)
            return result  # Already decoded to strings
            
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in keys(): {e}")
            self.is_connected = False
            return []
        except Exception as e:
            logger.error(f"Unexpected error in keys(): {e}")
            return []
    
    def ping(self) -> bool:
        """
        Ping Redis server to check connectivity.
        
        Returns:
            bool: True if ping successful
        """
        try:
            if not self.is_connected or not self.client:
                logger.warning("Redis not connected, attempting reconnection...")
                if not self.connect():
                    return False
            
            # Perform Redis PING operation
            result = self.client.ping()
            return bool(result)
            
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in ping(): {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error in ping(): {e}")
            return False
    
    def close(self):
        """Close Redis connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("SimpleRedisClient connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self.client = None
                self.is_connected = False