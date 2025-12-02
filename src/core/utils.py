#!/usr/bin/env python3
"""
utils.py: Common utility functions for the Agent-First Context System
"""

import re
from typing import Any, List, Optional


def sanitize_error_message(error_msg: str, sensitive_values: Optional[List[str]] = None) -> str:
    """
    Sanitize error messages to remove sensitive information

    Args:
        error_msg: The error message to sanitize
        sensitive_values: List of sensitive values to remove (passwords, tokens, etc.)

    Returns:
        Sanitized error message
    """
    if not error_msg:
        return error_msg

    sanitized = error_msg

    # First, handle provided sensitive values
    if sensitive_values:
        for value in sensitive_values:
            if not value or len(value) < 3:
                continue

            # Replace the value with asterisks
            pattern = re.escape(value)
            sanitized = re.sub(pattern, "***", sanitized, flags=re.IGNORECASE)

            # Also sanitize URL-encoded versions
            import urllib.parse

            encoded_value = urllib.parse.quote(value)
            if encoded_value != value:
                sanitized = re.sub(re.escape(encoded_value), "***", sanitized)

            # Base64 encoded version
            import base64

            try:
                b64_value = base64.b64encode(value.encode()).decode()
                sanitized = re.sub(re.escape(b64_value), "***", sanitized)
            except (AttributeError, UnicodeDecodeError, TypeError) as e:
                # Log the error but continue sanitization
                import logging

                logging.debug(f"Failed to encode value for sanitization: {e}")

    # Remove connection strings that might contain passwords
    # Match patterns like username:password@host
    sanitized = re.sub(r"://[^:/\s]+:[^@/\s]+@", "://***:***@", sanitized)

    # Remove auth headers and tokens
    # First handle specific auth header patterns
    sanitized = re.sub(
        r"Authorization:\s*Bearer\s+[A-Za-z0-9\-_]+(?:\.[A-Za-z0-9\-_]+)*",
        "Authorization: ***",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"Authorization:\s*Basic\s+[A-Za-z0-9+/=]+",
        "Authorization: ***",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Then handle general patterns with capture groups
    sanitized = re.sub(
        r'(password|api[_-]?key|token|secret|credential)[\s:=]+["\']?([^"\'\s]+)["\']?',
        r"\1: ***",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Handle patterns without capture groups
    # Match both JWT format (xxx.yyy.zzz) and simple tokens
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9\-_]+(?:\.[A-Za-z0-9\-_]+)*", "Bearer ***", sanitized)
    sanitized = re.sub(r"Basic\s+[A-Za-z0-9+/=]+", "Basic ***", sanitized)

    # Remove common password patterns in JSON/dict representations
    sanitized = re.sub(
        r'(["\']password["\']\s*:\s*["\'])[^"\']+(["\'])',
        r"\1***\2",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Remove database connection strings
    # Database connection strings with protocol capture
    sanitized = re.sub(
        r"(mongodb|postgres|postgresql|mysql|redis|neo4j)://[^@\s]+@[^\s]+",
        r"\1://***:***@***",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Bolt protocol without capture
    sanitized = re.sub(r"bolt\+s?://[^@\s]+@[^\s]+", "bolt://***:***@***", sanitized)

    return sanitized


def get_environment() -> str:
    """
    Detect the current environment

    Returns:
        Environment name: 'production', 'staging', or 'development'
    """
    import os

    # Check multiple environment variables
    env = os.getenv("ENVIRONMENT", os.getenv("ENV", os.getenv("NODE_ENV", "development"))).lower()

    # Map common variations
    if env in ["prod", "production"]:
        return "production"
    elif env in ["stage", "staging"]:
        return "staging"
    else:
        return "development"


def get_secure_connection_config(config: dict[str, Any], service: str) -> dict[str, Any]:
    """
    Get secure connection configuration with SSL/TLS options

    Args:
        config: Configuration dictionary
        service: Service name ('neo4j' or 'qdrant')

    Returns:
        Connection configuration with security options
    """
    service_config = config.get(service, {})
    environment = get_environment()

    # Force SSL in production unless explicitly disabled
    if environment == "production":
        default_ssl = service_config.get("ssl", True)
        if not default_ssl:
            import warnings

            warnings.warn(
                f"SSL is disabled for {service} in production environment. "
                "This is a security risk!",
                RuntimeWarning,
            )
    else:
        default_ssl = service_config.get("ssl", False)

    # Check environment variables first for Docker deployment
    # This allows Docker deployments to override config file settings
    import os
    
    if service == "neo4j":
        env_uri = os.getenv("NEO4J_URI")
        if env_uri:
            # Parse the URI to extract host and port
            import re
            uri_match = re.match(r"^[a-z+]+://([^:/]+):?(\d+)?", env_uri)
            if uri_match:
                host = uri_match.group(1)
                port = int(uri_match.group(2)) if uri_match.group(2) else 7687
            else:
                host = service_config.get("host", "localhost")
                port = service_config.get("port")
        else:
            host = service_config.get("host", "localhost")
            port = service_config.get("port")
    elif service == "redis":
        # Check REDIS_URL environment variable
        env_url = os.getenv("REDIS_URL")
        if env_url:
            # Parse redis://:password@host:port/db format (skip password for config)
            import re
            url_match = re.match(r"^redis://(?::([^@]+)@)?([^:/]+):?(\d+)?", env_url)
            if url_match:
                host = url_match.group(2)  # Host (after password)
                port = int(url_match.group(3)) if url_match.group(3) else 6379
            else:
                host = service_config.get("host", "localhost")
                port = service_config.get("port")
        else:
            host = service_config.get("host", "localhost")
            port = service_config.get("port")
    elif service == "qdrant":
        # Check QDRANT_URL environment variable
        env_url = os.getenv("QDRANT_URL")
        if env_url:
            # Parse http://host:port format
            import re
            url_match = re.match(r"^https?://([^:/]+):?(\d+)?", env_url)
            if url_match:
                host = url_match.group(1)
                port = int(url_match.group(2)) if url_match.group(2) else 6333
            else:
                host = service_config.get("host", "localhost")
                port = service_config.get("port")
        else:
            host = service_config.get("host", "localhost")
            port = service_config.get("port")
    else:
        # Default for other services
        host = service_config.get("host", "localhost")
        port = service_config.get("port")

    # Parse URI if present (common for Neo4j configuration in config file)
    uri = service_config.get("uri")
    if uri and not port:
        import re

        # Match protocol://host:port pattern
        uri_match = re.match(r"^[a-z+]+://([^:/]+):?(\d+)?", uri)
        if uri_match:
            host = uri_match.group(1)
            if uri_match.group(2):
                port = int(uri_match.group(2))

    # Set default ports if not specified
    if not port:
        if service == "neo4j":
            port = 7687
        elif service == "qdrant":
            port = 6333
        elif service == "redis":
            port = 6379
        else:
            port = None

    secure_config = {
        "host": host,
        "port": port,
        "ssl": default_ssl,
        "verify_ssl": service_config.get("verify_ssl", True),
        "timeout": service_config.get("timeout", 30),
        "environment": environment,
    }

    # Add SSL certificate paths if provided
    if service_config.get("ssl_cert_path"):
        secure_config["ssl_cert_path"] = service_config["ssl_cert_path"]
    if service_config.get("ssl_key_path"):
        secure_config["ssl_key_path"] = service_config["ssl_key_path"]
    if service_config.get("ssl_ca_path"):
        secure_config["ssl_ca_path"] = service_config["ssl_ca_path"]

    return secure_config
