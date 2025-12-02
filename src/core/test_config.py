#!/usr/bin/env python3
"""
test_config.py: Test configuration utilities for context storage system

This module provides test-friendly configuration loading and defaults
to enable proper test isolation without requiring actual config files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_test_defaults() -> Dict[str, Any]:
    """
    Load test defaults from centralized YAML configuration.
    
    Returns:
        Dictionary with test configuration loaded from config/test_defaults.yaml
    """
    # Find the project root directory
    current_dir = Path(__file__).parent
    while current_dir.parent != current_dir:
        config_file = current_dir / "config" / "test_defaults.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        current_dir = current_dir.parent
    
    # Fallback to hardcoded defaults if config file not found
    return _get_fallback_config()


def _get_fallback_config() -> Dict[str, Any]:
    """
    Fallback configuration when test_defaults.yaml is not available.
    
    Returns:
        Dictionary with fallback test configuration
    """
    return {
        "databases": {
            "neo4j": {
                "host": "localhost",
                "port": 7687,
                "database": "test_context",
                "username": "neo4j",
                "password": "test_password",
                "ssl": False,
            },
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_contexts",
                "dimensions": 384,
                "https": False,
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "database": 0,
                "password": None,
                "ssl": False,
            },
        },
        "storage": {"base_path": "/tmp/test_storage", "cache_ttl": 3600, "max_cache_size": 100},
        "embedding": {"model": "all-MiniLM-L6-v2", "cache_embeddings": True, "batch_size": 32},
        "security": {"auth_enabled": False, "ssl_enabled": False},
    }


def get_test_config() -> Dict[str, Any]:
    """
    Get default test configuration, loaded from centralized config file.

    Returns:
        Dictionary with test configuration for all storage backends
    """
    config_data = load_test_defaults()
    
    # Convert to legacy format for backward compatibility
    return {
        "neo4j": config_data["databases"]["neo4j"],
        "qdrant": config_data["databases"]["qdrant"],
        "redis": config_data["databases"]["redis"],
        "storage": config_data["storage"],
        "embedding": config_data["embedding"],
        "security": config_data["security"],
    }


def get_minimal_config() -> Dict[str, Any]:
    """
    Get minimal configuration for basic testing.

    Returns:
        Minimal configuration dictionary
    """
    config_data = load_test_defaults()
    databases = config_data["databases"]
    
    return {
        "neo4j": {"host": databases["neo4j"]["host"], "port": databases["neo4j"]["port"]},
        "qdrant": {"host": databases["qdrant"]["host"], "port": databases["qdrant"]["port"]},
        "redis": {"host": databases["redis"]["host"], "port": databases["redis"]["port"]},
    }


def merge_configs(
    base_config: Dict[str, Any], override_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge configuration dictionaries, with overrides taking precedence.

    Args:
        base_config: Base configuration dictionary
        override_config: Configuration overrides

    Returns:
        Merged configuration dictionary
    """
    if not override_config:
        return base_config.copy()

    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_test_credentials(service: str = None) -> Dict[str, Any]:
    """
    Get test credentials from centralized configuration.
    
    Args:
        service: Specific service to get credentials for (neo4j, redis, qdrant)
        
    Returns:
        Dictionary with test credentials
    """
    config_data = load_test_defaults()
    credentials = config_data.get("test_credentials", {})
    
    if service:
        return credentials.get(service, {})
    return credentials


def get_mock_responses(service: str = None) -> Dict[str, Any]:
    """
    Get mock responses for testing from centralized configuration.
    
    Args:
        service: Specific service to get mock responses for
        
    Returns:
        Dictionary with mock responses
    """
    config_data = load_test_defaults()
    mock_responses = config_data.get("mock_responses", {})
    
    if service:
        return mock_responses.get(service, {})
    return mock_responses


def get_test_data(data_type: str = None) -> Dict[str, Any]:
    """
    Get test data from centralized configuration.
    
    Args:
        data_type: Specific type of test data (sample_contexts, sample_vectors, etc.)
        
    Returns:
        Dictionary with test data
    """
    config_data = load_test_defaults()
    test_data = config_data.get("test_data", {})
    
    if data_type:
        return test_data.get(data_type, [])
    return test_data


def get_alternative_config(config_name: str) -> Dict[str, Any]:
    """
    Get alternative configuration for specific test scenarios.
    
    Args:
        config_name: Name of alternative config (integration_test, performance_test, ssl_test)
        
    Returns:
        Dictionary with alternative configuration
    """
    config_data = load_test_defaults()
    alt_configs = config_data.get("alternative_configs", {})
    
    if config_name in alt_configs:
        # Merge with base configuration
        base_config = get_test_config()
        return merge_configs(base_config, alt_configs[config_name])
    
    return get_test_config()  # Return default if not found


def get_database_url(service: str, ssl: bool = False) -> str:
    """
    Generate database URL for testing.
    
    Args:
        service: Database service (neo4j, qdrant, redis)
        ssl: Whether to use SSL/TLS
        
    Returns:
        Formatted database URL
    """
    config_data = load_test_defaults()
    databases = config_data["databases"]
    
    if service not in databases:
        raise ValueError(f"Unknown service: {service}")
    
    db_config = databases[service]
    
    if service == "neo4j":
        template = db_config.get("ssl_uri_template" if ssl else "uri_template", "bolt://{host}:{port}")
        return template.format(host=db_config["host"], port=db_config["port"])
    elif service == "qdrant":
        template = db_config.get("ssl_url_template" if ssl else "url_template", "http://{host}:{port}")
        return template.format(host=db_config["host"], port=db_config["port"])
    elif service == "redis":
        template = db_config.get("url_template", "redis://{host}:{port}")
        return template.format(host=db_config["host"], port=db_config["port"])
    
    raise ValueError(f"URL generation not supported for service: {service}")
