#!/usr/bin/env python3
"""
Central configuration module for context-store.
Provides consistent configuration values across all components.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Central configuration management for context-store."""

    # Embedding configuration - SPRINT 11: Fixed dimension drift (was 1536)
    EMBEDDING_DIMENSIONS = 384  # REQUIRED: Must be 384 for v1.0 compliance
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    EMBEDDING_BATCH_SIZE = 100
    EMBEDDING_MAX_RETRIES = 3

    # Database configuration defaults
    NEO4J_DEFAULT_PORT = 7687
    QDRANT_DEFAULT_PORT = 6333
    REDIS_DEFAULT_PORT = 6379

    # Connection pool settings
    CONNECTION_POOL_MIN_SIZE = 5
    CONNECTION_POOL_MAX_SIZE = 20
    CONNECTION_POOL_TIMEOUT = 30

    # Rate limiting configuration
    RATE_LIMIT_REQUESTS_PER_MINUTE = 60
    RATE_LIMIT_BURST_SIZE = 10

    # Security settings
    MAX_QUERY_LENGTH = 10000
    QUERY_TIMEOUT_SECONDS = 30
    ALLOWED_CYPHER_OPERATIONS = [
        "MATCH",
        "WITH",
        "RETURN",
        "WHERE",
        "ORDER BY",
        "LIMIT",
        "SKIP",
    ]
    FORBIDDEN_CYPHER_OPERATIONS = [
        "CREATE",
        "DELETE",
        "SET",
        "REMOVE",
        "MERGE",
        "DROP",
        "DETACH",
        "FOREACH",
    ]

    # Cache settings
    CACHE_TTL_SECONDS = 3600
    CACHE_MAX_SIZE = 1000

    # Semantic cache settings (Phase 1: S3 Paraphrase Robustness)
    SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
    SEMANTIC_CACHE_QUANTIZATION_PRECISION = int(os.getenv("SEMANTIC_CACHE_PRECISION", "1"))
    SEMANTIC_CACHE_EMBEDDING_PREFIX_LENGTH = int(os.getenv("SEMANTIC_CACHE_PREFIX_LENGTH", "32"))

    # Multi-Query Expansion settings (Phase 2: S3 Paraphrase Robustness)
    MQE_ENABLED = os.getenv("MQE_ENABLED", "true").lower() == "true"
    MQE_NUM_PARAPHRASES = int(os.getenv("MQE_NUM_PARAPHRASES", "2"))
    MQE_APPLY_FIELD_BOOSTS = os.getenv("MQE_APPLY_FIELD_BOOSTS", "true").lower() == "true"

    # Search enhancement settings (Phase 3: S3 Paraphrase Robustness)
    SEARCH_ENHANCEMENTS_ENABLED = os.getenv("SEARCH_ENHANCEMENTS_ENABLED", "true").lower() == "true"
    SEARCH_ENHANCEMENT_EXACT_MATCH = os.getenv("SEARCH_ENHANCEMENT_EXACT_MATCH", "true").lower() == "true"
    SEARCH_ENHANCEMENT_TYPE_WEIGHTING = os.getenv("SEARCH_ENHANCEMENT_TYPE_WEIGHTING", "true").lower() == "true"
    SEARCH_ENHANCEMENT_RECENCY_DECAY = os.getenv("SEARCH_ENHANCEMENT_RECENCY_DECAY", "true").lower() == "true"
    SEARCH_ENHANCEMENT_TECHNICAL_BOOST = os.getenv("SEARCH_ENHANCEMENT_TECHNICAL_BOOST", "true").lower() == "true"

    # Query normalization settings (Phase 4: S3 Paraphrase Robustness)
    QUERY_NORMALIZATION_ENABLED = os.getenv("QUERY_NORMALIZATION_ENABLED", "true").lower() == "true"
    QUERY_NORMALIZATION_CONFIDENCE_THRESHOLD = float(os.getenv("QUERY_NORMALIZATION_CONFIDENCE", "0.5"))

    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file. Defaults to .ctxrc.yaml

        Returns:
            Configuration dictionary
        """
        if config_path is None:
            config_path = os.getenv("CONTEXT_STORE_CONFIG", ".ctxrc.yaml")

        config_file = Path(config_path)
        if not config_file.exists():
            # Return default configuration
            return cls.get_defaults()

        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                defaults = cls.get_defaults()
                return cls._deep_merge(defaults, config)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get default configuration values.

        Returns:
            Default configuration dictionary
        """
        return {
            "embedding": {
                "dimensions": cls.EMBEDDING_DIMENSIONS,
                "model": cls.EMBEDDING_MODEL,
                "batch_size": cls.EMBEDDING_BATCH_SIZE,
                "max_retries": cls.EMBEDDING_MAX_RETRIES,
            },
            "databases": {
                "neo4j": {
                    "port": cls.NEO4J_DEFAULT_PORT,
                    "connection_pool": {
                        "min_size": cls.CONNECTION_POOL_MIN_SIZE,
                        "max_size": cls.CONNECTION_POOL_MAX_SIZE,
                        "timeout": cls.CONNECTION_POOL_TIMEOUT,
                    },
                },
                "qdrant": {
                    "port": cls.QDRANT_DEFAULT_PORT,
                    "connection_pool": {
                        "min_size": cls.CONNECTION_POOL_MIN_SIZE,
                        "max_size": cls.CONNECTION_POOL_MAX_SIZE,
                    },
                },
                "redis": {
                    "port": cls.REDIS_DEFAULT_PORT,
                    "connection_pool": {
                        "min_size": cls.CONNECTION_POOL_MIN_SIZE,
                        "max_size": cls.CONNECTION_POOL_MAX_SIZE,
                    },
                },
            },
            "security": {
                "max_query_length": cls.MAX_QUERY_LENGTH,
                "query_timeout": cls.QUERY_TIMEOUT_SECONDS,
                "allowed_operations": cls.ALLOWED_CYPHER_OPERATIONS,
                "forbidden_operations": cls.FORBIDDEN_CYPHER_OPERATIONS,
            },
            "rate_limiting": {
                "requests_per_minute": cls.RATE_LIMIT_REQUESTS_PER_MINUTE,
                "burst_size": cls.RATE_LIMIT_BURST_SIZE,
            },
            "cache": {
                "ttl_seconds": cls.CACHE_TTL_SECONDS,
                "max_size": cls.CACHE_MAX_SIZE,
            },
            "logging": {
                "level": cls.LOG_LEVEL,
                "format": cls.LOG_FORMAT,
            },
        }

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def validate_configuration(cls, config: Dict[str, Any]) -> bool:
        """Validate configuration values.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate embedding dimensions
        if "embedding" in config:
            dims = config["embedding"].get("dimensions")
            if dims and dims not in [384, 768, 1536, 3072]:
                raise ConfigurationError(f"Invalid embedding dimensions: {dims}")

        # Validate ports
        for db in ["neo4j", "qdrant", "redis"]:
            if db in config.get("databases", {}):
                port = config["databases"][db].get("port")
                if port and (not isinstance(port, int) or port < 1 or port > 65535):
                    raise ConfigurationError(f"Invalid port for {db}: {port}")

        # Validate connection pool settings
        for db in config.get("databases", {}).values():
            if "connection_pool" in db:
                pool = db["connection_pool"]
                min_size = pool.get("min_size", 1)
                max_size = pool.get("max_size", 10)
                if min_size > max_size:
                    raise ConfigurationError("Connection pool min_size cannot exceed max_size")

        return True


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


# Singleton instance
_config_instance: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """Get the global configuration instance.

    Returns:
        Configuration dictionary
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config.load_from_file()
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Reload configuration from file.

    Args:
        config_path: Optional path to configuration file

    Returns:
        New configuration dictionary
    """
    global _config_instance
    _config_instance = Config.load_from_file(config_path)
    return _config_instance
