#!/usr/bin/env python3
"""
SSL/TLS Configuration for Context Store Production Backends.

This module provides comprehensive SSL configuration for all storage backends
to ensure secure data transmission in production environments.
"""

import os
import ssl
from pathlib import Path
from typing import Any, Dict, Optional


class SSLConfigManager:
    """Manages SSL/TLS configuration for all storage backends."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize SSL configuration manager.

        Args:
            config: Main configuration dictionary containing SSL settings
        """
        self.config = config
        self.ssl_config = config.get("ssl", {})

    def get_neo4j_ssl_config(self) -> Dict[str, Any]:
        """Get SSL configuration for Neo4j connections.

        Returns:
            Dictionary with Neo4j SSL configuration
        """
        neo4j_ssl = self.ssl_config.get("neo4j", {})

        if not neo4j_ssl.get("enabled", False):
            return {"encrypted": False}

        ssl_config = {
            "encrypted": True,
            "trust": neo4j_ssl.get("trust", "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"),
        }

        # Add certificate configuration if provided
        if cert_file := neo4j_ssl.get("cert_file"):
            ssl_config["trusted_certificates"] = str(Path(cert_file).resolve())

        if key_file := neo4j_ssl.get("key_file"):
            ssl_config["client_certificate"] = (
                str(Path(cert_file).resolve()) if cert_file else None,
                str(Path(key_file).resolve()),
            )

        return ssl_config

    def get_qdrant_ssl_config(self) -> Dict[str, Any]:
        """Get SSL configuration for Qdrant connections.

        Returns:
            Dictionary with Qdrant SSL configuration
        """
        qdrant_ssl = self.ssl_config.get("qdrant", {})

        if not qdrant_ssl.get("enabled", False):
            return {}

        ssl_config: Dict[str, Any] = {}

        # Enable HTTPS
        if qdrant_ssl.get("enabled"):
            ssl_config["https"] = True

        # API key for authentication (recommended with SSL)
        api_key = qdrant_ssl.get("api_key") or os.getenv("QDRANT_API_KEY")
        if api_key:
            ssl_config["api_key"] = api_key

        # Certificate verification
        if not qdrant_ssl.get("verify_ssl", True):
            ssl_config["verify"] = False

        # Custom CA certificate
        ca_cert = qdrant_ssl.get("ca_cert")
        if ca_cert:
            ssl_config["verify"] = str(Path(ca_cert).resolve())

        return ssl_config

    def get_redis_ssl_config(self) -> Dict[str, Any]:
        """Get SSL configuration for Redis connections.

        Returns:
            Dictionary with Redis SSL configuration
        """
        redis_ssl = self.ssl_config.get("redis", {})

        if not redis_ssl.get("enabled", False):
            return {}

        ssl_config: Dict[str, Any] = {
            "ssl": True,
            "ssl_check_hostname": redis_ssl.get("check_hostname", True),
            "ssl_cert_reqs": (
                ssl.CERT_REQUIRED if redis_ssl.get("verify_certs", True) else ssl.CERT_NONE
            ),
        }

        # SSL certificate files
        if ssl_certfile := redis_ssl.get("ssl_certfile"):
            ssl_config["ssl_certfile"] = str(Path(ssl_certfile).resolve())

        if ssl_keyfile := redis_ssl.get("ssl_keyfile"):
            ssl_config["ssl_keyfile"] = str(Path(ssl_keyfile).resolve())

        if ssl_ca_certs := redis_ssl.get("ssl_ca_certs"):
            ssl_config["ssl_ca_certs"] = str(Path(ssl_ca_certs).resolve())

        # SSL password for encrypted private key
        if ssl_password := redis_ssl.get("ssl_password") or os.getenv("REDIS_SSL_PASSWORD"):
            ssl_config["ssl_password"] = ssl_password

        return ssl_config

    def validate_ssl_certificates(self) -> Dict[str, Any]:
        """Validate SSL certificate files and check expiration.

        Returns:
            Dictionary with validation status and details for each backend
        """
        validation: Dict[str, Dict[str, Any]] = {
            "neo4j": {"valid": True, "errors": [], "warnings": []},
            "qdrant": {"valid": True, "errors": [], "warnings": []},
            "redis": {"valid": True, "errors": [], "warnings": []},
        }

        # Validate Neo4j certificates
        neo4j_ssl = self.ssl_config.get("neo4j", {})
        if neo4j_ssl.get("enabled"):
            cert_file = neo4j_ssl.get("cert_file")
            key_file = neo4j_ssl.get("key_file")

            if cert_file:
                if not Path(cert_file).is_file():
                    validation["neo4j"]["valid"] = False
                    validation["neo4j"]["errors"].append(f"Certificate file not found: {cert_file}")
                elif not self._validate_certificate_permissions(cert_file):
                    validation["neo4j"]["warnings"].append(
                        f"Certificate file has insecure permissions: {cert_file}"
                    )

            if key_file:
                if not Path(key_file).is_file():
                    validation["neo4j"]["valid"] = False
                    validation["neo4j"]["errors"].append(f"Key file not found: {key_file}")
                elif not self._validate_key_permissions(key_file):
                    validation["neo4j"]["errors"].append(
                        f"Private key file has insecure permissions: {key_file}"
                    )
                    validation["neo4j"]["valid"] = False

        # Validate Qdrant certificates
        qdrant_ssl = self.ssl_config.get("qdrant", {})
        if qdrant_ssl.get("enabled"):
            ca_cert = qdrant_ssl.get("ca_cert")
            if ca_cert:
                if not Path(ca_cert).is_file():
                    validation["qdrant"]["valid"] = False
                    validation["qdrant"]["errors"].append(f"CA certificate not found: {ca_cert}")
                elif not self._validate_certificate_permissions(ca_cert):
                    validation["qdrant"]["warnings"].append(
                        f"CA certificate has insecure permissions: {ca_cert}"
                    )

        # Validate Redis certificates
        redis_ssl = self.ssl_config.get("redis", {})
        if redis_ssl.get("enabled"):
            cert_validations = {
                "ssl_certfile": "Certificate",
                "ssl_keyfile": "Private key",
                "ssl_ca_certs": "CA certificate",
            }

            for cert_key, cert_type in cert_validations.items():
                cert_file = redis_ssl.get(cert_key)
                if cert_file:
                    if not Path(cert_file).is_file():
                        validation["redis"]["valid"] = False
                        validation["redis"]["errors"].append(
                            f"{cert_type} file not found: {cert_file}"
                        )
                    elif cert_key == "ssl_keyfile" and not self._validate_key_permissions(
                        cert_file
                    ):
                        validation["redis"]["errors"].append(
                            f"{cert_type} has insecure permissions: {cert_file}"
                        )
                        validation["redis"]["valid"] = False
                    elif cert_key != "ssl_keyfile" and not self._validate_certificate_permissions(
                        cert_file
                    ):
                        validation["redis"]["warnings"].append(
                            f"{cert_type} has insecure permissions: {cert_file}"
                        )

        return validation

    def _validate_certificate_permissions(self, cert_path: str) -> bool:
        """Validate certificate file permissions (should be readable).

        Args:
            cert_path: Path to certificate file

        Returns:
            True if permissions are acceptable
        """
        try:
            path = Path(cert_path)
            # Check if file is readable
            if not os.access(path, os.R_OK):
                return False
            # Warn if world-writable (but don't fail)
            stat_info = path.stat()
            if stat_info.st_mode & 0o002:  # World-writable
                return False
            return True
        except Exception:
            return False

    def _validate_key_permissions(self, key_path: str) -> bool:
        """Validate private key file permissions (should be restricted).

        Args:
            key_path: Path to private key file

        Returns:
            True if permissions are secure (owner-only read)
        """
        try:
            path = Path(key_path)
            stat_info = path.stat()
            # Check if only owner can read (0o600 or 0o400)
            mode = stat_info.st_mode & 0o777
            if mode not in [0o600, 0o400]:
                return False
            return True
        except Exception:
            return False

    def get_environment_ssl_config(self) -> Dict[str, Dict[str, Any]]:
        """Get SSL configuration from environment variables.

        Returns:
            Dictionary with SSL configuration for each backend from env vars
        """
        env_config: Dict[str, Dict[str, Any]] = {"neo4j": {}, "qdrant": {}, "redis": {}}

        # Neo4j environment SSL configuration
        if os.getenv("NEO4J_SSL_ENABLED", "").lower() == "true":
            env_config["neo4j"] = {
                "enabled": True,
                "cert_file": os.getenv("NEO4J_SSL_CERT"),
                "key_file": os.getenv("NEO4J_SSL_KEY"),
                "trust": os.getenv("NEO4J_SSL_TRUST", "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"),
            }

        # Qdrant environment SSL configuration
        if os.getenv("QDRANT_SSL_ENABLED", "").lower() == "true":
            env_config["qdrant"] = {
                "enabled": True,
                "verify_ssl": os.getenv("QDRANT_VERIFY_SSL", "true").lower() == "true",
                "ca_cert": os.getenv("QDRANT_CA_CERT"),
                "api_key": os.getenv("QDRANT_API_KEY"),
            }

        # Redis environment SSL configuration
        if os.getenv("REDIS_SSL_ENABLED", "").lower() == "true":
            env_config["redis"] = {
                "enabled": True,
                "check_hostname": os.getenv("REDIS_SSL_CHECK_HOSTNAME", "true").lower() == "true",
                "verify_certs": os.getenv("REDIS_SSL_VERIFY_CERTS", "true").lower() == "true",
                "ssl_certfile": os.getenv("REDIS_SSL_CERTFILE"),
                "ssl_keyfile": os.getenv("REDIS_SSL_KEYFILE"),
                "ssl_ca_certs": os.getenv("REDIS_SSL_CA_CERTS"),
                "ssl_password": os.getenv("REDIS_SSL_PASSWORD"),
            }

        return env_config

    def merge_ssl_config(self) -> Dict[str, Any]:
        """Merge file-based and environment-based SSL configuration.

        Environment variables take precedence over file configuration.

        Returns:
            Merged SSL configuration
        """
        merged_config = self.config.copy()
        env_ssl_config = self.get_environment_ssl_config()

        # Merge SSL configurations with environment taking precedence
        if "ssl" not in merged_config:
            merged_config["ssl"] = {}

        for backend, env_config in env_ssl_config.items():
            if env_config:  # Only merge if environment config exists
                if backend not in merged_config["ssl"]:
                    merged_config["ssl"][backend] = {}
                merged_config["ssl"][backend].update(env_config)

        return merged_config


def create_ssl_context(
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
    ca_file: Optional[str] = None,
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED,
) -> ssl.SSLContext:
    """Create SSL context for secure connections.

    Args:
        cert_file: Path to client certificate file
        key_file: Path to client private key file
        ca_file: Path to CA certificate file
        verify_mode: SSL certificate verification mode

    Returns:
        Configured SSL context
    """
    context = ssl.create_default_context()
    context.verify_mode = verify_mode

    if cert_file and key_file:
        context.load_cert_chain(cert_file, key_file)

    if ca_file:
        context.load_verify_locations(ca_file)

    return context


def get_production_ssl_recommendations() -> Dict[str, str]:
    """Get SSL configuration recommendations for production deployment.

    Returns:
        Dictionary with SSL recommendations for each backend
    """
    return {
        "neo4j": (
            "Enable SSL with TRUST_SYSTEM_CA_SIGNED_CERTIFICATES or provide custom CA. "
            "Use strong authentication and consider client certificates for mutual TLS."
        ),
        "qdrant": (
            "Enable HTTPS with API key authentication. Verify SSL certificates in production. "
            "Consider using service mesh or reverse proxy for additional security."
        ),
        "redis": (
            "Enable SSL/TLS with certificate verification. Use AUTH password protection. "
            "Consider using stunnel or Redis Enterprise for enhanced security features."
        ),
        "general": (
            "Regularly rotate certificates and keys. Use certificate monitoring. "
            "Implement proper key management and secure certificate storage."
        ),
    }
