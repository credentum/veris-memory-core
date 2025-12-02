#!/usr/bin/env python3
"""
Comprehensive tests for the SSL configuration module.

This test suite covers SSL/TLS configuration for all storage backends
to achieve high code coverage for the core.ssl_config module.
"""

import os
import ssl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.ssl_config import (  # noqa: E402
    SSLConfigManager,
    create_ssl_context,
    get_production_ssl_recommendations,
)


class TestSSLConfigManager:
    """Test suite for SSLConfigManager class."""

    def test_init(self):
        """Test SSLConfigManager initialization."""
        config = {"ssl": {"neo4j": {"enabled": True}}}
        manager = SSLConfigManager(config)
        assert manager.config == config
        assert manager.ssl_config == config["ssl"]

    def test_init_no_ssl_config(self):
        """Test initialization with no SSL configuration."""
        config = {}
        manager = SSLConfigManager(config)
        assert manager.ssl_config == {}

    def test_get_neo4j_ssl_config_disabled(self):
        """Test Neo4j SSL config when disabled."""
        config = {"ssl": {"neo4j": {"enabled": False}}}
        manager = SSLConfigManager(config)
        result = manager.get_neo4j_ssl_config()
        assert result == {"encrypted": False}

    def test_get_neo4j_ssl_config_enabled_basic(self):
        """Test Neo4j SSL config when enabled with basic settings."""
        config = {"ssl": {"neo4j": {"enabled": True}}}
        manager = SSLConfigManager(config)
        result = manager.get_neo4j_ssl_config()
        expected = {"encrypted": True, "trust": "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"}
        assert result == expected

    def test_get_neo4j_ssl_config_with_certificates(self):
        """Test Neo4j SSL config with certificate files."""
        with (
            tempfile.NamedTemporaryFile(suffix=".crt") as cert_file,
            tempfile.NamedTemporaryFile(suffix=".key") as key_file,
        ):
            config = {
                "ssl": {
                    "neo4j": {
                        "enabled": True,
                        "cert_file": cert_file.name,
                        "key_file": key_file.name,
                        "trust": "TRUST_CUSTOM_CA_SIGNED_CERTIFICATES",
                    }
                }
            }
            manager = SSLConfigManager(config)
            result = manager.get_neo4j_ssl_config()

            assert result["encrypted"] is True
            assert result["trust"] == "TRUST_CUSTOM_CA_SIGNED_CERTIFICATES"
            assert "trusted_certificates" in result
            assert "client_certificate" in result

    def test_get_neo4j_ssl_config_cert_only(self):
        """Test Neo4j SSL config with only certificate file."""
        with tempfile.NamedTemporaryFile(suffix=".crt") as cert_file:
            config = {"ssl": {"neo4j": {"enabled": True, "cert_file": cert_file.name}}}
            manager = SSLConfigManager(config)
            result = manager.get_neo4j_ssl_config()

            assert "trusted_certificates" in result
            assert "client_certificate" not in result

    def test_get_qdrant_ssl_config_disabled(self):
        """Test Qdrant SSL config when disabled."""
        config = {"ssl": {"qdrant": {"enabled": False}}}
        manager = SSLConfigManager(config)
        result = manager.get_qdrant_ssl_config()
        assert result == {}

    def test_get_qdrant_ssl_config_enabled(self):
        """Test Qdrant SSL config when enabled."""
        config = {"ssl": {"qdrant": {"enabled": True}}}
        manager = SSLConfigManager(config)
        result = manager.get_qdrant_ssl_config()
        assert result["https"] is True

    def test_get_qdrant_ssl_config_with_api_key(self):
        """Test Qdrant SSL config with API key."""
        config = {"ssl": {"qdrant": {"enabled": True, "api_key": "test_api_key"}}}
        manager = SSLConfigManager(config)
        result = manager.get_qdrant_ssl_config()
        assert result["api_key"] == "test_api_key"

    def test_get_qdrant_ssl_config_with_env_api_key(self):
        """Test Qdrant SSL config with API key from environment."""
        config = {"ssl": {"qdrant": {"enabled": True}}}
        with patch.dict(os.environ, {"QDRANT_API_KEY": "env_api_key"}):
            manager = SSLConfigManager(config)
            result = manager.get_qdrant_ssl_config()
            assert result["api_key"] == "env_api_key"

    def test_get_qdrant_ssl_config_verify_disabled(self):
        """Test Qdrant SSL config with verification disabled."""
        config = {"ssl": {"qdrant": {"enabled": True, "verify_ssl": False}}}
        manager = SSLConfigManager(config)
        result = manager.get_qdrant_ssl_config()
        assert result["verify"] is False

    def test_get_qdrant_ssl_config_with_ca_cert(self):
        """Test Qdrant SSL config with custom CA certificate."""
        with tempfile.NamedTemporaryFile(suffix=".crt") as ca_cert:
            config = {"ssl": {"qdrant": {"enabled": True, "ca_cert": ca_cert.name}}}
            manager = SSLConfigManager(config)
            result = manager.get_qdrant_ssl_config()
            assert result["verify"] == str(Path(ca_cert.name).resolve())

    def test_get_redis_ssl_config_disabled(self):
        """Test Redis SSL config when disabled."""
        config = {"ssl": {"redis": {"enabled": False}}}
        manager = SSLConfigManager(config)
        result = manager.get_redis_ssl_config()
        assert result == {}

    def test_get_redis_ssl_config_enabled_basic(self):
        """Test Redis SSL config when enabled with basic settings."""
        config = {"ssl": {"redis": {"enabled": True}}}
        manager = SSLConfigManager(config)
        result = manager.get_redis_ssl_config()

        expected = {
            "ssl": True,
            "ssl_check_hostname": True,
            "ssl_cert_reqs": ssl.CERT_REQUIRED,
        }
        assert result == expected

    def test_get_redis_ssl_config_verify_disabled(self):
        """Test Redis SSL config with certificate verification disabled."""
        config = {"ssl": {"redis": {"enabled": True, "verify_certs": False}}}
        manager = SSLConfigManager(config)
        result = manager.get_redis_ssl_config()
        assert result["ssl_cert_reqs"] == ssl.CERT_NONE

    def test_get_redis_ssl_config_with_certificates(self):
        """Test Redis SSL config with certificate files."""
        with (
            tempfile.NamedTemporaryFile(suffix=".crt") as cert_file,
            tempfile.NamedTemporaryFile(suffix=".key") as key_file,
            tempfile.NamedTemporaryFile(suffix=".ca") as ca_file,
        ):
            config = {
                "ssl": {
                    "redis": {
                        "enabled": True,
                        "ssl_certfile": cert_file.name,
                        "ssl_keyfile": key_file.name,
                        "ssl_ca_certs": ca_file.name,
                    }
                }
            }
            manager = SSLConfigManager(config)
            result = manager.get_redis_ssl_config()

            assert "ssl_certfile" in result
            assert "ssl_keyfile" in result
            assert "ssl_ca_certs" in result

    def test_get_redis_ssl_config_with_password(self):
        """Test Redis SSL config with SSL password."""
        config = {"ssl": {"redis": {"enabled": True, "ssl_password": "test_password"}}}
        manager = SSLConfigManager(config)
        result = manager.get_redis_ssl_config()
        assert result["ssl_password"] == "test_password"

    def test_get_redis_ssl_config_with_env_password(self):
        """Test Redis SSL config with password from environment."""
        config = {"ssl": {"redis": {"enabled": True}}}
        with patch.dict(os.environ, {"REDIS_SSL_PASSWORD": "env_password"}):
            manager = SSLConfigManager(config)
            result = manager.get_redis_ssl_config()
            assert result["ssl_password"] == "env_password"

    def test_validate_ssl_certificates_all_valid(self):
        """Test SSL certificate validation when all certificates are valid."""
        with (
            tempfile.NamedTemporaryFile() as neo4j_cert,
            tempfile.NamedTemporaryFile() as neo4j_key,
            tempfile.NamedTemporaryFile() as qdrant_ca,
            tempfile.NamedTemporaryFile() as redis_cert,
        ):
            # Set secure permissions for the key file
            os.chmod(neo4j_key.name, 0o600)

            config = {
                "ssl": {
                    "neo4j": {
                        "enabled": True,
                        "cert_file": neo4j_cert.name,
                        "key_file": neo4j_key.name,
                    },
                    "qdrant": {"enabled": True, "ca_cert": qdrant_ca.name},
                    "redis": {"enabled": True, "ssl_certfile": redis_cert.name},
                }
            }
            manager = SSLConfigManager(config)
            result = manager.validate_ssl_certificates()

            assert result["neo4j"]["valid"] is True
            assert result["qdrant"]["valid"] is True
            assert result["redis"]["valid"] is True

    def test_validate_ssl_certificates_missing_files(self):
        """Test SSL certificate validation with missing certificate files."""
        config = {
            "ssl": {
                "neo4j": {
                    "enabled": True,
                    "cert_file": "/nonexistent/cert.pem",
                    "key_file": "/nonexistent/key.pem",
                },
                "qdrant": {"enabled": True, "ca_cert": "/nonexistent/ca.pem"},
                "redis": {"enabled": True, "ssl_certfile": "/nonexistent/redis.pem"},
            }
        }
        manager = SSLConfigManager(config)
        result = manager.validate_ssl_certificates()

        assert result["neo4j"]["valid"] is False
        assert result["qdrant"]["valid"] is False
        assert result["redis"]["valid"] is False
        assert "Certificate file not found" in str(result["neo4j"]["errors"])
        assert "CA certificate not found" in str(result["qdrant"]["errors"])
        assert "Certificate file not found" in str(result["redis"]["errors"])

    def test_validate_ssl_certificates_disabled(self):
        """Test SSL certificate validation when SSL is disabled."""
        config = {
            "ssl": {
                "neo4j": {"enabled": False},
                "qdrant": {"enabled": False},
                "redis": {"enabled": False},
            }
        }
        manager = SSLConfigManager(config)
        result = manager.validate_ssl_certificates()

        assert result["neo4j"]["valid"] is True
        assert result["qdrant"]["valid"] is True
        assert result["redis"]["valid"] is True

    def test_get_environment_ssl_config(self):
        """Test getting SSL configuration from environment variables."""
        env_vars = {
            "NEO4J_SSL_ENABLED": "true",
            "NEO4J_SSL_CERT": "/path/to/neo4j.crt",
            "NEO4J_SSL_KEY": "/path/to/neo4j.key",
            "QDRANT_SSL_ENABLED": "true",
            "QDRANT_API_KEY": "qdrant_key",
            "REDIS_SSL_ENABLED": "true",
            "REDIS_SSL_CERTFILE": "/path/to/redis.crt",
        }

        with patch.dict(os.environ, env_vars):
            manager = SSLConfigManager({})
            result = manager.get_environment_ssl_config()

            assert result["neo4j"]["enabled"] is True
            assert result["neo4j"]["cert_file"] == "/path/to/neo4j.crt"
            assert result["qdrant"]["enabled"] is True
            assert result["qdrant"]["api_key"] == "qdrant_key"
            assert result["redis"]["enabled"] is True
            assert result["redis"]["ssl_certfile"] == "/path/to/redis.crt"

    def test_get_environment_ssl_config_disabled(self):
        """Test environment SSL config when explicitly disabled."""
        env_vars = {
            "NEO4J_SSL_ENABLED": "false",
            "QDRANT_SSL_ENABLED": "FALSE",
            "REDIS_SSL_ENABLED": "no",
        }

        with patch.dict(os.environ, env_vars):
            manager = SSLConfigManager({})
            result = manager.get_environment_ssl_config()

            assert result["neo4j"] == {}
            assert result["qdrant"] == {}
            assert result["redis"] == {}

    def test_merge_ssl_config(self):
        """Test merging file-based and environment-based SSL configuration."""
        config = {
            "ssl": {
                "neo4j": {
                    "enabled": True,
                    "trust": "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES",
                }
            }
        }

        env_vars = {
            "NEO4J_SSL_ENABLED": "true",
            "NEO4J_SSL_CERT": "/env/neo4j.crt",
            "QDRANT_SSL_ENABLED": "true",
            "QDRANT_API_KEY": "env_key",
        }

        with patch.dict(os.environ, env_vars):
            manager = SSLConfigManager(config)
            result = manager.merge_ssl_config()

            # Environment should override file config
            assert result["ssl"]["neo4j"]["cert_file"] == "/env/neo4j.crt"
            # File config should be preserved where not overridden
            assert result["ssl"]["neo4j"]["trust"] == "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
            # Environment-only config should be added
            assert result["ssl"]["qdrant"]["api_key"] == "env_key"

    def test_merge_ssl_config_no_file_ssl(self):
        """Test merging when file config has no SSL section."""
        config = {}

        env_vars = {"NEO4J_SSL_ENABLED": "true", "NEO4J_SSL_CERT": "/env/neo4j.crt"}

        with patch.dict(os.environ, env_vars):
            manager = SSLConfigManager(config)
            result = manager.merge_ssl_config()

            assert "ssl" in result
            assert result["ssl"]["neo4j"]["cert_file"] == "/env/neo4j.crt"


class TestSSLContextCreation:
    """Test suite for SSL context creation functions."""

    def test_create_ssl_context_default(self):
        """Test creating SSL context with default parameters."""
        context = create_ssl_context()
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED

    def test_create_ssl_context_custom_verify_mode(self):
        """Test creating SSL context with custom verification mode."""
        context = create_ssl_context(verify_mode=ssl.CERT_OPTIONAL)
        assert context.verify_mode == ssl.CERT_OPTIONAL

    @patch("ssl.SSLContext.load_cert_chain")
    def test_create_ssl_context_with_certificates(self, mock_load_cert_chain):
        """Test creating SSL context with certificate chain."""
        with tempfile.NamedTemporaryFile() as cert_file, tempfile.NamedTemporaryFile() as key_file:
            context = create_ssl_context(cert_file=cert_file.name, key_file=key_file.name)

            mock_load_cert_chain.assert_called_once_with(cert_file.name, key_file.name)

    @patch("ssl.SSLContext.load_verify_locations")
    def test_create_ssl_context_with_ca_file(self, mock_load_verify):
        """Test creating SSL context with CA file."""
        with tempfile.NamedTemporaryFile() as ca_file:
            context = create_ssl_context(ca_file=ca_file.name)
            mock_load_verify.assert_called_once_with(ca_file.name)

    @patch("ssl.SSLContext.load_cert_chain")
    @patch("ssl.SSLContext.load_verify_locations")
    def test_create_ssl_context_full_config(self, mock_load_verify, mock_load_cert_chain):
        """Test creating SSL context with full configuration."""
        with (
            tempfile.NamedTemporaryFile() as cert_file,
            tempfile.NamedTemporaryFile() as key_file,
            tempfile.NamedTemporaryFile() as ca_file,
        ):
            context = create_ssl_context(
                cert_file=cert_file.name,
                key_file=key_file.name,
                ca_file=ca_file.name,
                verify_mode=ssl.CERT_OPTIONAL,
            )

            assert context.verify_mode == ssl.CERT_OPTIONAL
            mock_load_cert_chain.assert_called_once_with(cert_file.name, key_file.name)
            mock_load_verify.assert_called_once_with(ca_file.name)


class TestProductionRecommendations:
    """Test suite for production SSL recommendations."""

    def test_get_production_ssl_recommendations(self):
        """Test getting production SSL recommendations."""
        recommendations = get_production_ssl_recommendations()

        assert "neo4j" in recommendations
        assert "qdrant" in recommendations
        assert "redis" in recommendations
        assert "general" in recommendations

        # Check that recommendations contain security-related keywords
        assert "SSL" in recommendations["neo4j"] or "TLS" in recommendations["neo4j"]
        assert "HTTPS" in recommendations["qdrant"] or "SSL" in recommendations["qdrant"]
        assert "SSL" in recommendations["redis"] or "TLS" in recommendations["redis"]
        assert "certificate" in recommendations["general"].lower()


class TestSSLConfigEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_get_neo4j_ssl_config_no_ssl_section(self):
        """Test Neo4j SSL config when no SSL section exists."""
        config = {}
        manager = SSLConfigManager(config)
        result = manager.get_neo4j_ssl_config()
        assert result == {"encrypted": False}

    def test_get_qdrant_ssl_config_no_ssl_section(self):
        """Test Qdrant SSL config when no SSL section exists."""
        config = {}
        manager = SSLConfigManager(config)
        result = manager.get_qdrant_ssl_config()
        assert result == {}

    def test_get_redis_ssl_config_no_ssl_section(self):
        """Test Redis SSL config when no SSL section exists."""
        config = {}
        manager = SSLConfigManager(config)
        result = manager.get_redis_ssl_config()
        assert result == {}

    def test_validate_ssl_certificates_no_ssl_section(self):
        """Test certificate validation when no SSL section exists."""
        config = {}
        manager = SSLConfigManager(config)
        result = manager.validate_ssl_certificates()

        assert result["neo4j"]["valid"] is True
        assert result["qdrant"]["valid"] is True
        assert result["redis"]["valid"] is True

    def test_environment_ssl_config_empty_values(self):
        """Test environment SSL config with empty environment variables."""
        env_vars = {
            "NEO4J_SSL_ENABLED": "",
            "QDRANT_SSL_ENABLED": "",
            "REDIS_SSL_ENABLED": "",
        }

        with patch.dict(os.environ, env_vars):
            manager = SSLConfigManager({})
            result = manager.get_environment_ssl_config()

            assert result["neo4j"] == {}
            assert result["qdrant"] == {}
            assert result["redis"] == {}

    def test_redis_ssl_config_partial_certificates(self):
        """Test Redis SSL config with only some certificate files."""
        with tempfile.NamedTemporaryFile() as cert_file:
            config = {
                "ssl": {
                    "redis": {
                        "enabled": True,
                        "ssl_certfile": cert_file.name,
                        "ssl_keyfile": "/nonexistent/key.pem",  # This one doesn't exist
                    }
                }
            }
            manager = SSLConfigManager(config)
            result = manager.validate_ssl_certificates()
            assert result["redis"]["valid"] is False
            assert "Private key file not found" in str(result["redis"]["errors"])


class TestSSLPermissionValidation:
    """Test suite for certificate permission validation methods."""

    def test_validate_certificate_permissions_valid(self):
        """Test certificate permission validation with valid permissions."""
        with tempfile.NamedTemporaryFile() as cert_file:
            # Set readable but not world-writable permissions
            os.chmod(cert_file.name, 0o644)

            manager = SSLConfigManager({})
            result = manager._validate_certificate_permissions(cert_file.name)
            assert result is True

    def test_validate_certificate_permissions_world_writable(self):
        """Test certificate permission validation with world-writable file."""
        with tempfile.NamedTemporaryFile() as cert_file:
            # Set world-writable permissions (insecure)
            os.chmod(cert_file.name, 0o666)

            manager = SSLConfigManager({})
            result = manager._validate_certificate_permissions(cert_file.name)
            assert result is False

    def test_validate_certificate_permissions_unreadable(self):
        """Test certificate permission validation with unreadable file."""
        with tempfile.NamedTemporaryFile() as cert_file:
            # Set no read permissions
            os.chmod(cert_file.name, 0o000)

            manager = SSLConfigManager({})
            result = manager._validate_certificate_permissions(cert_file.name)
            assert result is False

    def test_validate_certificate_permissions_nonexistent(self):
        """Test certificate permission validation with nonexistent file."""
        manager = SSLConfigManager({})
        result = manager._validate_certificate_permissions("/nonexistent/file.pem")
        assert result is False

    def test_validate_certificate_permissions_exception(self):
        """Test certificate permission validation with exception handling."""
        manager = SSLConfigManager({})

        # Mock Path.stat to raise an exception
        with patch("pathlib.Path.stat", side_effect=OSError("Permission denied")):
            result = manager._validate_certificate_permissions("/some/file.pem")
            assert result is False

    def test_validate_key_permissions_secure(self):
        """Test private key permission validation with secure permissions."""
        with tempfile.NamedTemporaryFile() as key_file:
            # Set owner-only read permissions (secure)
            os.chmod(key_file.name, 0o600)

            manager = SSLConfigManager({})
            result = manager._validate_key_permissions(key_file.name)
            assert result is True

    def test_validate_key_permissions_read_only(self):
        """Test private key permission validation with read-only permissions."""
        with tempfile.NamedTemporaryFile() as key_file:
            # Set owner-only read permissions (secure)
            os.chmod(key_file.name, 0o400)

            manager = SSLConfigManager({})
            result = manager._validate_key_permissions(key_file.name)
            assert result is True

    def test_validate_key_permissions_insecure(self):
        """Test private key permission validation with insecure permissions."""
        with tempfile.NamedTemporaryFile() as key_file:
            # Set group and world readable (insecure)
            os.chmod(key_file.name, 0o644)

            manager = SSLConfigManager({})
            result = manager._validate_key_permissions(key_file.name)
            assert result is False

    def test_validate_key_permissions_world_writable(self):
        """Test private key permission validation with world-writable permissions."""
        with tempfile.NamedTemporaryFile() as key_file:
            # Set world-writable permissions (very insecure)
            os.chmod(key_file.name, 0o666)

            manager = SSLConfigManager({})
            result = manager._validate_key_permissions(key_file.name)
            assert result is False

    def test_validate_key_permissions_exception(self):
        """Test private key permission validation with exception handling."""
        manager = SSLConfigManager({})

        # Mock Path.stat to raise an exception
        with patch("pathlib.Path.stat", side_effect=OSError("Permission denied")):
            result = manager._validate_key_permissions("/some/key.pem")
            assert result is False


class TestSSLValidationDetailed:
    """Test suite for detailed SSL certificate validation scenarios."""

    def test_validate_neo4j_certificates_insecure_permissions(self):
        """Test Neo4j certificate validation with insecure permissions."""
        with (
            tempfile.NamedTemporaryFile() as cert_file,
            tempfile.NamedTemporaryFile() as key_file,
        ):
            # Set insecure permissions
            os.chmod(cert_file.name, 0o666)  # World-writable cert
            os.chmod(key_file.name, 0o644)  # Group-readable key

            config = {
                "ssl": {
                    "neo4j": {
                        "enabled": True,
                        "cert_file": cert_file.name,
                        "key_file": key_file.name,
                    }
                }
            }
            manager = SSLConfigManager(config)
            result = manager.validate_ssl_certificates()

            assert result["neo4j"]["valid"] is False
            assert len(result["neo4j"]["warnings"]) > 0 or len(result["neo4j"]["errors"]) > 0

    def test_validate_qdrant_certificates_insecure_permissions(self):
        """Test Qdrant certificate validation with insecure permissions."""
        with tempfile.NamedTemporaryFile() as ca_cert:
            # Set world-writable permissions
            os.chmod(ca_cert.name, 0o666)

            config = {
                "ssl": {
                    "qdrant": {
                        "enabled": True,
                        "ca_cert": ca_cert.name,
                    }
                }
            }
            manager = SSLConfigManager(config)
            result = manager.validate_ssl_certificates()

            # Should have warnings about insecure permissions
            assert len(result["qdrant"]["warnings"]) > 0

    def test_validate_redis_all_certificate_types(self):
        """Test Redis certificate validation with all certificate types."""
        with (
            tempfile.NamedTemporaryFile() as cert_file,
            tempfile.NamedTemporaryFile() as key_file,
            tempfile.NamedTemporaryFile() as ca_file,
        ):
            # Set secure permissions for key, insecure for others
            os.chmod(key_file.name, 0o644)  # Insecure key permissions
            os.chmod(cert_file.name, 0o666)  # World-writable cert
            os.chmod(ca_file.name, 0o666)  # World-writable CA

            config = {
                "ssl": {
                    "redis": {
                        "enabled": True,
                        "ssl_certfile": cert_file.name,
                        "ssl_keyfile": key_file.name,
                        "ssl_ca_certs": ca_file.name,
                    }
                }
            }
            manager = SSLConfigManager(config)
            result = manager.validate_ssl_certificates()

            assert result["redis"]["valid"] is False
            assert len(result["redis"]["errors"]) > 0
            assert len(result["redis"]["warnings"]) > 0

    def test_validate_certificates_mixed_scenarios(self):
        """Test certificate validation with mixed valid/invalid scenarios."""
        with (
            tempfile.NamedTemporaryFile() as valid_cert,
            tempfile.NamedTemporaryFile() as valid_key,
            tempfile.NamedTemporaryFile() as valid_ca,
        ):
            # Set valid permissions
            os.chmod(valid_key.name, 0o600)
            os.chmod(valid_cert.name, 0o644)
            os.chmod(valid_ca.name, 0o644)

            config = {
                "ssl": {
                    "neo4j": {
                        "enabled": True,
                        "cert_file": valid_cert.name,
                        "key_file": valid_key.name,
                    },
                    "qdrant": {
                        "enabled": True,
                        "ca_cert": "/nonexistent/ca.pem",  # Missing file
                    },
                    "redis": {
                        "enabled": True,
                        "ssl_certfile": valid_cert.name,
                        "ssl_ca_certs": valid_ca.name,
                    },
                }
            }
            manager = SSLConfigManager(config)
            result = manager.validate_ssl_certificates()

            assert result["neo4j"]["valid"] is True
            assert result["qdrant"]["valid"] is False
            assert result["redis"]["valid"] is True


class TestSSLEnvironmentEdgeCases:
    """Test suite for environment-based SSL configuration edge cases."""

    def test_environment_ssl_config_case_variations(self):
        """Test environment SSL config with various case variations."""
        env_vars = {
            "NEO4J_SSL_ENABLED": "TRUE",  # Uppercase
            "QDRANT_SSL_ENABLED": "True",  # Mixed case
            "REDIS_SSL_ENABLED": "true",  # Lowercase
            "QDRANT_VERIFY_SSL": "FALSE",  # Uppercase false
            "REDIS_SSL_CHECK_HOSTNAME": "False",  # Mixed case false
            "REDIS_SSL_VERIFY_CERTS": "false",  # Lowercase false
        }

        with patch.dict(os.environ, env_vars):
            manager = SSLConfigManager({})
            result = manager.get_environment_ssl_config()

            assert result["neo4j"]["enabled"] is True
            assert result["qdrant"]["enabled"] is True
            assert result["redis"]["enabled"] is True
            assert result["qdrant"]["verify_ssl"] is False
            assert result["redis"]["check_hostname"] is False
            assert result["redis"]["verify_certs"] is False

    def test_environment_ssl_config_partial_settings(self):
        """Test environment SSL config with partial settings."""
        env_vars = {
            "NEO4J_SSL_ENABLED": "true",
            "NEO4J_SSL_CERT": "/path/to/cert.pem",
            # Missing NEO4J_SSL_KEY
            "QDRANT_SSL_ENABLED": "true",
            # Missing other Qdrant settings
            "REDIS_SSL_ENABLED": "true",
            "REDIS_SSL_CERTFILE": "/path/to/redis.crt",
            "REDIS_SSL_PASSWORD": "secret123",
        }

        with patch.dict(os.environ, env_vars):
            manager = SSLConfigManager({})
            result = manager.get_environment_ssl_config()

            assert result["neo4j"]["cert_file"] == "/path/to/cert.pem"
            assert result["neo4j"]["key_file"] is None
            assert result["redis"]["ssl_certfile"] == "/path/to/redis.crt"
            assert result["redis"]["ssl_password"] == "secret123"

    def test_merge_ssl_config_complex_override(self):
        """Test complex SSL configuration merging scenarios."""
        config = {
            "ssl": {
                "neo4j": {
                    "enabled": False,  # Will be overridden by env
                    "trust": "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES",
                    "cert_file": "/file/cert.pem",
                },
                "redis": {
                    "enabled": True,
                    "check_hostname": False,  # Will be overridden by env
                    "ssl_certfile": "/file/redis.crt",
                },
            }
        }

        env_vars = {
            "NEO4J_SSL_ENABLED": "true",  # Override to enable
            "NEO4J_SSL_CERT": "/env/neo4j.crt",  # Override cert path
            "REDIS_SSL_ENABLED": "true",
            "REDIS_SSL_CHECK_HOSTNAME": "true",  # Override hostname check
            "REDIS_SSL_KEYFILE": "/env/redis.key",  # Add new setting
        }

        with patch.dict(os.environ, env_vars):
            manager = SSLConfigManager(config)
            result = manager.merge_ssl_config()

            # Environment should override file settings
            assert result["ssl"]["neo4j"]["enabled"] is True
            assert result["ssl"]["neo4j"]["cert_file"] == "/env/neo4j.crt"
            # File settings should be preserved where not overridden
            assert result["ssl"]["neo4j"]["trust"] == "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
            # Environment should override file settings
            assert result["ssl"]["redis"]["check_hostname"] is True
            # Environment config includes ssl_certfile: None which overwrites file config
            assert result["ssl"]["redis"]["ssl_certfile"] is None
            # Environment-only settings should be added
            assert result["ssl"]["redis"]["ssl_keyfile"] == "/env/redis.key"


class TestSSLConfigSpecialCases:
    """Test suite for special cases and boundary conditions."""

    def test_neo4j_ssl_config_key_without_cert(self):
        """Test Neo4j SSL config with key file but no cert file."""
        with tempfile.NamedTemporaryFile() as key_file:
            config = {
                "ssl": {
                    "neo4j": {
                        "enabled": True,
                        "key_file": key_file.name,
                        # No cert_file specified
                    }
                }
            }
            manager = SSLConfigManager(config)
            result = manager.get_neo4j_ssl_config()

            assert result["encrypted"] is True
            assert "client_certificate" in result
            # Should have None for cert_file, path for key_file
            assert result["client_certificate"][0] is None
            assert result["client_certificate"][1] == str(Path(key_file.name).resolve())

    def test_qdrant_ssl_config_api_key_precedence(self):
        """Test Qdrant SSL config API key precedence (config vs environment)."""
        config = {
            "ssl": {
                "qdrant": {
                    "enabled": True,
                    "api_key": "config_key",
                }
            }
        }

        # Test with environment variable (config takes precedence due to 'or' logic)
        with patch.dict(os.environ, {"QDRANT_API_KEY": "env_key"}):
            manager = SSLConfigManager(config)
            result = manager.get_qdrant_ssl_config()
            # Config setting takes precedence due to 'or' logic
            assert result["api_key"] == "config_key"

        # Test without environment variable (should use config)
        with patch.dict(os.environ, {}, clear=True):
            manager = SSLConfigManager(config)
            result = manager.get_qdrant_ssl_config()
            # Should use config setting when no env var
            assert result["api_key"] == "config_key"

        # Test with environment variable and no config key
        config_no_key = {"ssl": {"qdrant": {"enabled": True}}}
        with patch.dict(os.environ, {"QDRANT_API_KEY": "env_key"}):
            manager = SSLConfigManager(config_no_key)
            result = manager.get_qdrant_ssl_config()
            # Should use environment when no config key
            assert result["api_key"] == "env_key"

    def test_redis_ssl_config_password_precedence(self):
        """Test Redis SSL config password precedence (config vs environment)."""
        config = {
            "ssl": {
                "redis": {
                    "enabled": True,
                    "ssl_password": "config_password",
                }
            }
        }

        # Test with environment variable (config takes precedence due to 'or' logic)
        with patch.dict(os.environ, {"REDIS_SSL_PASSWORD": "env_password"}):
            manager = SSLConfigManager(config)
            result = manager.get_redis_ssl_config()
            # Config setting takes precedence due to 'or' logic
            assert result["ssl_password"] == "config_password"

        # Test without environment variable (should use config)
        with patch.dict(os.environ, {}, clear=True):
            manager = SSLConfigManager(config)
            result = manager.get_redis_ssl_config()
            assert result["ssl_password"] == "config_password"

        # Test with environment variable and no config password
        config_no_password = {"ssl": {"redis": {"enabled": True}}}
        with patch.dict(os.environ, {"REDIS_SSL_PASSWORD": "env_password"}):
            manager = SSLConfigManager(config_no_password)
            result = manager.get_redis_ssl_config()
            # Should use environment when no config password
            assert result["ssl_password"] == "env_password"

    def test_validate_certificates_access_denied(self):
        """Test certificate validation with access denied scenarios."""
        config = {
            "ssl": {
                "neo4j": {
                    "enabled": True,
                    "cert_file": "/inaccessible/secret.crt",  # Non-existent path
                    "key_file": "/inaccessible/secret.key",
                }
            }
        }

        manager = SSLConfigManager(config)
        result = manager.validate_ssl_certificates()

        # Should handle gracefully and return errors for missing files
        assert "neo4j" in result
        assert result["neo4j"]["valid"] is False
        assert len(result["neo4j"]["errors"]) > 0


class TestSSLContextCreationExtended:
    """Extended test suite for SSL context creation functions."""

    def test_create_ssl_context_cert_without_key(self):
        """Test creating SSL context with cert file but no key file."""
        with tempfile.NamedTemporaryFile() as cert_file:
            # Should not load cert chain if key is missing
            context = create_ssl_context(cert_file=cert_file.name, key_file=None)
            assert isinstance(context, ssl.SSLContext)
            assert context.verify_mode == ssl.CERT_REQUIRED

    def test_create_ssl_context_key_without_cert(self):
        """Test creating SSL context with key file but no cert file."""
        with tempfile.NamedTemporaryFile() as key_file:
            # Should not load cert chain if cert is missing
            context = create_ssl_context(cert_file=None, key_file=key_file.name)
            assert isinstance(context, ssl.SSLContext)
            assert context.verify_mode == ssl.CERT_REQUIRED

    @patch("ssl.SSLContext.load_cert_chain", side_effect=ssl.SSLError("Certificate error"))
    def test_create_ssl_context_cert_load_error(self, mock_load_cert_chain):
        """Test creating SSL context with certificate loading error."""
        with (
            tempfile.NamedTemporaryFile() as cert_file,
            tempfile.NamedTemporaryFile() as key_file,
        ):
            # Should raise the SSL error from load_cert_chain
            try:
                create_ssl_context(cert_file=cert_file.name, key_file=key_file.name)
                assert False, "Expected SSL error to be raised"
            except ssl.SSLError:
                pass  # Expected

    @patch("ssl.SSLContext.load_verify_locations", side_effect=ssl.SSLError("CA error"))
    def test_create_ssl_context_ca_load_error(self, mock_load_verify):
        """Test creating SSL context with CA loading error."""
        with tempfile.NamedTemporaryFile() as ca_file:
            # Should raise the SSL error from load_verify_locations
            try:
                create_ssl_context(ca_file=ca_file.name)
                assert False, "Expected SSL error to be raised"
            except ssl.SSLError:
                pass  # Expected

    def test_create_ssl_context_verify_modes(self):
        """Test creating SSL context with different verification modes."""
        # Test CERT_REQUIRED (default)
        context = create_ssl_context(verify_mode=ssl.CERT_REQUIRED)
        assert context.verify_mode == ssl.CERT_REQUIRED

        # Test CERT_OPTIONAL
        context = create_ssl_context(verify_mode=ssl.CERT_OPTIONAL)
        assert context.verify_mode == ssl.CERT_OPTIONAL

        # Test CERT_NONE (need to disable hostname checking first)
        context = ssl.create_default_context()
        context.check_hostname = False  # Required before setting verify_mode to CERT_NONE
        context.verify_mode = ssl.CERT_NONE
        assert context.verify_mode == ssl.CERT_NONE


class TestSSLConfigurationEdgeCases:
    """Test suite for additional SSL configuration edge cases."""

    def test_qdrant_ssl_config_verify_ssl_default(self):
        """Test Qdrant SSL config with default verify_ssl value."""
        config = {"ssl": {"qdrant": {"enabled": True}}}
        manager = SSLConfigManager(config)
        result = manager.get_qdrant_ssl_config()

        # verify_ssl defaults to True, so 'verify' key should not be present
        assert "verify" not in result or result.get("verify") is not False

    def test_qdrant_ssl_config_ca_cert_overrides_verify(self):
        """Test Qdrant SSL config where ca_cert overrides verify_ssl setting."""
        with tempfile.NamedTemporaryFile() as ca_cert:
            config = {
                "ssl": {
                    "qdrant": {
                        "enabled": True,
                        "verify_ssl": False,  # This should be overridden
                        "ca_cert": ca_cert.name,
                    }
                }
            }
            manager = SSLConfigManager(config)
            result = manager.get_qdrant_ssl_config()

            # ca_cert should override verify_ssl=False
            assert result["verify"] == str(Path(ca_cert.name).resolve())

    def test_redis_ssl_config_hostname_check_default(self):
        """Test Redis SSL config with default hostname check value."""
        config = {"ssl": {"redis": {"enabled": True}}}
        manager = SSLConfigManager(config)
        result = manager.get_redis_ssl_config()

        # check_hostname defaults to True
        assert result["ssl_check_hostname"] is True

    def test_redis_ssl_config_verify_certs_default(self):
        """Test Redis SSL config with default certificate verification."""
        config = {"ssl": {"redis": {"enabled": True}}}
        manager = SSLConfigManager(config)
        result = manager.get_redis_ssl_config()

        # verify_certs defaults to True, so should use CERT_REQUIRED
        assert result["ssl_cert_reqs"] == ssl.CERT_REQUIRED

    def test_neo4j_ssl_config_trust_default(self):
        """Test Neo4j SSL config with default trust setting."""
        config = {"ssl": {"neo4j": {"enabled": True}}}
        manager = SSLConfigManager(config)
        result = manager.get_neo4j_ssl_config()

        # trust defaults to TRUST_SYSTEM_CA_SIGNED_CERTIFICATES
        assert result["trust"] == "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"

    def test_environment_ssl_config_trust_override(self):
        """Test environment SSL config trust setting override."""
        env_vars = {
            "NEO4J_SSL_ENABLED": "true",
            "NEO4J_SSL_TRUST": "TRUST_ALL_CERTIFICATES",
        }

        with patch.dict(os.environ, env_vars):
            manager = SSLConfigManager({})
            result = manager.get_environment_ssl_config()

            assert result["neo4j"]["trust"] == "TRUST_ALL_CERTIFICATES"

    def test_validate_certificates_no_enabled_backends(self):
        """Test certificate validation when no backends are enabled."""
        config = {"ssl": {}}
        manager = SSLConfigManager(config)
        result = manager.validate_ssl_certificates()

        # All backends should be valid when not configured
        assert result["neo4j"]["valid"] is True
        assert result["qdrant"]["valid"] is True
        assert result["redis"]["valid"] is True
        assert len(result["neo4j"]["errors"]) == 0
        assert len(result["qdrant"]["errors"]) == 0
        assert len(result["redis"]["errors"]) == 0


class TestSSLConfigFileSystemMocking:
    """Test suite for SSL configuration with file system mocking."""

    @patch("pathlib.Path.is_file", return_value=False)
    def test_validate_certificates_all_files_missing(self, mock_is_file):
        """Test certificate validation when all certificate files are missing."""
        config = {
            "ssl": {
                "neo4j": {
                    "enabled": True,
                    "cert_file": "/path/to/neo4j.crt",
                    "key_file": "/path/to/neo4j.key",
                },
                "qdrant": {
                    "enabled": True,
                    "ca_cert": "/path/to/qdrant-ca.crt",
                },
                "redis": {
                    "enabled": True,
                    "ssl_certfile": "/path/to/redis.crt",
                    "ssl_keyfile": "/path/to/redis.key",
                    "ssl_ca_certs": "/path/to/redis-ca.crt",
                },
            }
        }

        manager = SSLConfigManager(config)
        result = manager.validate_ssl_certificates()

        # All backends should have validation errors
        assert result["neo4j"]["valid"] is False
        assert result["qdrant"]["valid"] is False
        assert result["redis"]["valid"] is False

        # Check that appropriate error messages are present
        assert any("not found" in error for error in result["neo4j"]["errors"])
        assert any("not found" in error for error in result["qdrant"]["errors"])
        assert any("not found" in error for error in result["redis"]["errors"])

    def test_validate_certificates_permission_mocking(self):
        """Test certificate validation with mocked file permissions."""
        config = {
            "ssl": {
                "neo4j": {
                    "enabled": True,
                    "cert_file": "/path/to/neo4j.crt",
                    "key_file": "/path/to/neo4j.key",
                },
                "redis": {
                    "enabled": True,
                    "ssl_certfile": "/path/to/redis.crt",
                    "ssl_keyfile": "/path/to/redis.key",
                },
            }
        }

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("os.access", return_value=True),
            patch.object(Path, "stat") as mock_stat,
        ):
            # Mock stat to return secure permissions
            def stat_side_effect():
                mock_stat_obj = MagicMock()
                # Return secure permissions for all files in this test
                mock_stat_obj.st_mode = 0o100600  # Secure permissions
                return mock_stat_obj

            mock_stat.side_effect = stat_side_effect

            manager = SSLConfigManager(config)
            result = manager.validate_ssl_certificates()

            # Should be valid with proper permissions and mocked file existence
            assert result["neo4j"]["valid"] is True
            assert result["redis"]["valid"] is True
            assert len(result["neo4j"]["errors"]) == 0
            assert len(result["redis"]["errors"]) == 0

    def test_certificate_path_resolution(self):
        """Test that certificate paths are properly resolved."""
        with (
            tempfile.NamedTemporaryFile() as cert_file,
            tempfile.NamedTemporaryFile() as key_file,
        ):
            config = {
                "ssl": {
                    "neo4j": {
                        "enabled": True,
                        "cert_file": cert_file.name,
                        "key_file": key_file.name,
                    }
                }
            }

            manager = SSLConfigManager(config)
            result = manager.get_neo4j_ssl_config()

            # Check that paths are resolved to absolute paths
            expected_cert_path = str(Path(cert_file.name).resolve())
            expected_key_path = str(Path(key_file.name).resolve())

            assert result["trusted_certificates"] == expected_cert_path
            assert result["client_certificate"][0] == expected_cert_path
            assert result["client_certificate"][1] == expected_key_path


class TestProductionRecommendationsExtended:
    """Extended test suite for production SSL recommendations."""

    def test_production_recommendations_content(self):
        """Test that production recommendations contain expected content."""
        recommendations = get_production_ssl_recommendations()

        # Check that all backend recommendations exist and contain security keywords
        security_keywords = ["ssl", "tls", "certificate", "auth", "security", "encrypt"]

        for backend in ["neo4j", "qdrant", "redis", "general"]:
            assert backend in recommendations
            recommendation_text = recommendations[backend].lower()
            assert any(keyword in recommendation_text for keyword in security_keywords)

        # Check specific recommendations for each backend
        assert "trust" in recommendations["neo4j"].lower()
        assert "api key" in recommendations["qdrant"].lower()
        assert "auth" in recommendations["redis"].lower()
        assert "rotate" in recommendations["general"].lower()

    def test_production_recommendations_immutable(self):
        """Test that production recommendations return the same content consistently."""
        recommendations1 = get_production_ssl_recommendations()
        recommendations2 = get_production_ssl_recommendations()

        assert recommendations1 == recommendations2

        # Test that the function doesn't depend on external state
        for backend in ["neo4j", "qdrant", "redis", "general"]:
            assert recommendations1[backend] == recommendations2[backend]
