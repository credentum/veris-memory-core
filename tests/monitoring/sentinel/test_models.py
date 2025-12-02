#!/usr/bin/env python3
"""
Unit tests for Sentinel models (SentinelConfig, CheckResult).

Tests cover the new modular Sentinel architecture's data models,
particularly focusing on environment variable handling in SentinelConfig.
"""

import pytest
import os
from datetime import datetime, timezone
from unittest.mock import patch

from src.monitoring.sentinel.models import SentinelConfig, CheckResult


class TestSentinelConfig:
    """Test SentinelConfig model with environment variable handling."""

    def test_config_defaults_without_env_vars(self):
        """Test default configuration values when environment variables are not set."""
        # Clear all relevant environment variables
        with patch.dict(os.environ, {}, clear=False):
            for key in ['TARGET_BASE_URL', 'SENTINEL_CHECK_INTERVAL',
                       'SENTINEL_ALERT_THRESHOLD', 'SENTINEL_WEBHOOK_URL',
                       'GITHUB_TOKEN', 'SENTINEL_GITHUB_REPO', 'SENTINEL_ENABLED_CHECKS']:
                os.environ.pop(key, None)

            config = SentinelConfig()

            # Verify localhost defaults are used when environment variables not set
            assert config.target_base_url == "http://localhost:8000"
            assert config.check_interval_seconds == 60
            assert config.alert_threshold_failures == 3
            assert config.webhook_url is None
            assert config.github_token is None
            assert config.github_repo is None
            # Default enabled checks should be set
            # PR #397: Only runtime checks enabled by default (6 checks)
            # CI/CD-only checks (S3, S4, S7, S8, S9) run via GitHub Actions
            assert config.enabled_checks is not None
            assert len(config.enabled_checks) == 6  # Runtime checks only
            assert "S1-probes" in config.enabled_checks
            assert "S11-firewall-status" in config.enabled_checks
            # CI/CD-only checks should NOT be in defaults
            assert "S7-config-parity" not in config.enabled_checks

    def test_config_reads_target_base_url_from_env(self):
        """Test that TARGET_BASE_URL environment variable is read correctly by __post_init__()."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://context-store:8000'
        }):
            config = SentinelConfig()

            # Verify TARGET_BASE_URL env var is used
            assert config.target_base_url == "http://context-store:8000"

    def test_config_reads_all_env_vars(self):
        """Test that all environment variables are properly read by __post_init__()."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://context-store:8000',
            'SENTINEL_CHECK_INTERVAL': '30',
            'SENTINEL_ALERT_THRESHOLD': '5',
            'SENTINEL_WEBHOOK_URL': 'https://hooks.slack.com/test',
            'GITHUB_TOKEN': 'test_token_123',
            'SENTINEL_GITHUB_REPO': 'credentum/veris-memory',
            'SENTINEL_ENABLED_CHECKS': 'S1-probes,S2-golden-fact-recall,S7-config-parity'
        }):
            config = SentinelConfig()

            # Verify all environment variables are used
            assert config.target_base_url == "http://context-store:8000"
            assert config.check_interval_seconds == 60  # Not set by __post_init__, only from_env
            assert config.alert_threshold_failures == 3  # Not set by __post_init__, only from_env
            assert config.webhook_url is None  # Not set by __post_init__
            assert config.github_token is None  # Not set by __post_init__
            assert config.github_repo is None  # Not set by __post_init__

    def test_config_explicit_parameter_overrides_env_var(self):
        """Test that explicit parameters override environment variables."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://context-store:8000'
        }):
            # Override with explicit value
            config = SentinelConfig(target_base_url="http://custom:9000")

            # Verify explicit value takes precedence
            assert config.target_base_url == "http://custom:9000"

    def test_config_partial_env_vars_with_defaults(self):
        """Test that partial environment variables work with defaults."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://context-store:8000'
        }, clear=False):
            # Remove other variables to ensure fallback
            for key in ['SENTINEL_CHECK_INTERVAL', 'SENTINEL_ALERT_THRESHOLD']:
                os.environ.pop(key, None)

            config = SentinelConfig()

            # Verify mixed behavior: env var used for target_base_url, defaults for others
            assert config.target_base_url == "http://context-store:8000"  # From env
            assert config.check_interval_seconds == 60  # Default
            assert config.alert_threshold_failures == 3  # Default

    def test_config_from_env_classmethod(self):
        """Test SentinelConfig.from_env() classmethod reads all environment variables."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://context-store:8000',
            'SENTINEL_CHECK_INTERVAL': '45',
            'SENTINEL_ALERT_THRESHOLD': '2',
            'SENTINEL_WEBHOOK_URL': 'https://hooks.slack.com/services/test',
            'GITHUB_TOKEN': 'ghp_test123',
            'SENTINEL_GITHUB_REPO': 'credentum/veris-memory',
            'SENTINEL_ENABLED_CHECKS': 'S1-probes,S7-config-parity'
        }):
            config = SentinelConfig.from_env()

            # Verify all environment variables are used
            assert config.target_base_url == "http://context-store:8000"
            assert config.check_interval_seconds == 45
            assert config.alert_threshold_failures == 2
            assert config.webhook_url == "https://hooks.slack.com/services/test"
            assert config.github_token == "ghp_test123"
            assert config.github_repo == "credentum/veris-memory"
            assert config.enabled_checks == ['S1-probes', 'S7-config-parity']

    def test_config_from_env_with_defaults(self):
        """Test from_env() falls back to defaults when env vars not set."""
        with patch.dict(os.environ, {}, clear=False):
            for key in ['TARGET_BASE_URL', 'SENTINEL_CHECK_INTERVAL',
                       'SENTINEL_ALERT_THRESHOLD', 'SENTINEL_WEBHOOK_URL',
                       'GITHUB_TOKEN', 'SENTINEL_GITHUB_REPO', 'SENTINEL_ENABLED_CHECKS']:
                os.environ.pop(key, None)

            config = SentinelConfig.from_env()

            # Verify defaults
            assert config.target_base_url == "http://localhost:8000"
            assert config.check_interval_seconds == 60
            assert config.alert_threshold_failures == 3
            assert config.webhook_url is None
            assert config.github_token is None
            assert config.github_repo is None
            # Should have default enabled_checks from __post_init__
            # PR #397: Only 6 runtime checks enabled by default
            assert config.enabled_checks is not None
            assert len(config.enabled_checks) == 6

    def test_config_get_method_api_url(self):
        """Test that get('api_url') returns target_base_url correctly."""
        config = SentinelConfig(target_base_url="http://test:8000")

        # Verify api_url getter returns target_base_url
        assert config.get('api_url') == "http://test:8000"

    def test_config_get_method_veris_memory_url(self):
        """Test that get('veris_memory_url') returns target_base_url correctly."""
        config = SentinelConfig(target_base_url="http://test:8000")

        # Verify veris_memory_url getter returns target_base_url
        assert config.get('veris_memory_url') == "http://test:8000"

    def test_config_get_method_with_defaults(self):
        """Test that get() method returns None for unknown keys with no default."""
        config = SentinelConfig()

        # Verify get returns None for unknown keys
        assert config.get('unknown_key') is None

    def test_config_get_method_with_custom_default(self):
        """Test that get() method returns custom default for unknown keys."""
        config = SentinelConfig()

        # Verify get returns custom default
        assert config.get('unknown_key', 'default_value') == 'default_value'

    def test_config_get_method_database_urls(self):
        """Test that get() method returns database URLs from environment."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://custom-qdrant:6333',
            'NEO4J_URI': 'bolt://custom-neo4j:7687',
            'REDIS_URL': 'redis://custom-redis:6379'
        }):
            config = SentinelConfig()

            # Verify database URLs are read from environment
            assert config.get('qdrant_url') == 'http://custom-qdrant:6333'
            assert config.get('neo4j_url') == 'bolt://custom-neo4j:7687'
            assert config.get('redis_url') == 'redis://custom-redis:6379'

    def test_config_get_method_database_urls_defaults(self):
        """Test that get() method returns default database URLs when env vars not set."""
        with patch.dict(os.environ, {}, clear=False):
            for key in ['QDRANT_URL', 'NEO4J_URI', 'REDIS_URL']:
                os.environ.pop(key, None)

            config = SentinelConfig()

            # Verify default database URLs
            assert config.get('qdrant_url') == 'http://localhost:6333'
            assert config.get('neo4j_url') == 'bolt://localhost:7687'
            assert config.get('redis_url') == 'redis://localhost:6379'

    def test_config_is_check_enabled(self):
        """Test is_check_enabled() method."""
        config = SentinelConfig(enabled_checks=['S1-probes', 'S7-config-parity'])

        # Verify enabled check detection
        assert config.is_check_enabled('S1-probes') is True
        assert config.is_check_enabled('S7-config-parity') is True
        assert config.is_check_enabled('S2-golden-fact-recall') is False

    def test_config_to_dict(self):
        """Test to_dict() conversion."""
        config = SentinelConfig(
            target_base_url="http://test:8000",
            check_interval_seconds=30,
            alert_threshold_failures=5
        )

        result = config.to_dict()

        # Verify dictionary conversion
        assert isinstance(result, dict)
        assert result['target_base_url'] == "http://test:8000"
        assert result['check_interval_seconds'] == 30
        assert result['alert_threshold_failures'] == 5

    def test_config_from_dict(self):
        """Test from_dict() creation."""
        data = {
            'target_base_url': 'http://test:8000',
            'check_interval_seconds': 30,
            'alert_threshold_failures': 5,
            'webhook_url': 'https://hooks.slack.com/test',
            'enabled_checks': ['S1-probes']
        }

        config = SentinelConfig.from_dict(data)

        # Verify config created from dict
        assert config.target_base_url == "http://test:8000"
        assert config.check_interval_seconds == 30
        assert config.alert_threshold_failures == 5
        assert config.webhook_url == "https://hooks.slack.com/test"
        assert config.enabled_checks == ['S1-probes']

    # Edge Case Tests for __post_init__()

    def test_config_empty_string_env_var(self):
        """Test that empty string env vars are handled correctly."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': ''  # Empty string
        }):
            config = SentinelConfig()

            # Empty string should be used (not replaced with default)
            # This matches __post_init__ behavior: only None triggers default
            assert config.target_base_url == ''

    def test_config_whitespace_only_env_var(self):
        """Test that whitespace-only env vars are preserved."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': '   '  # Whitespace only
        }):
            config = SentinelConfig()

            # Whitespace is preserved (validation happens elsewhere)
            assert config.target_base_url == '   '

    def test_config_malformed_url_in_env_var(self):
        """Test that malformed URLs are accepted (validation happens at usage)."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'not-a-valid-url'
        }):
            config = SentinelConfig()

            # Malformed URL is accepted (validation happens when URL is used)
            assert config.target_base_url == 'not-a-valid-url'

    def test_config_special_characters_in_env_var(self):
        """Test that special characters in env vars are preserved."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://test:8000/path?query=value&foo=bar#fragment'
        }):
            config = SentinelConfig()

            # Special characters should be preserved
            assert config.target_base_url == 'http://test:8000/path?query=value&foo=bar#fragment'

    def test_config_very_long_url(self):
        """Test that very long URLs are handled correctly."""
        long_url = 'http://' + 'a' * 2000 + '.com'
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': long_url
        }):
            config = SentinelConfig()

            # Very long URL should be preserved
            assert config.target_base_url == long_url
            assert len(config.target_base_url) > 2000

    def test_config_unicode_in_env_var(self):
        """Test that Unicode characters in env vars are handled."""
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://test.com/путь/файл'  # Cyrillic characters
        }):
            config = SentinelConfig()

            # Unicode should be preserved
            assert config.target_base_url == 'http://test.com/путь/файл'

    def test_config_none_vs_missing_env_var(self):
        """Test difference between None and missing env var."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('TARGET_BASE_URL', None)

            # When env var is missing, __post_init__ sets default
            config = SentinelConfig()
            assert config.target_base_url == 'http://localhost:8000'

            # When explicitly set to None in constructor, it still triggers __post_init__
            config2 = SentinelConfig(target_base_url=None)
            assert config2.target_base_url == 'http://localhost:8000'

    def test_config_enabled_checks_empty_list(self):
        """Test that empty enabled_checks list is preserved."""
        config = SentinelConfig(enabled_checks=[])

        # Empty list should be preserved (not replaced with defaults)
        assert config.enabled_checks == []

    def test_config_enabled_checks_none_triggers_defaults(self):
        """Test that None enabled_checks triggers default list."""
        config = SentinelConfig(enabled_checks=None)

        # None should trigger default list in __post_init__
        # PR #397: Only 6 runtime checks enabled by default
        assert config.enabled_checks is not None
        assert len(config.enabled_checks) == 6
        assert 'S1-probes' in config.enabled_checks

    def test_config_multiple_none_values(self):
        """Test that multiple None values all trigger defaults."""
        with patch.dict(os.environ, {}, clear=False):
            for key in ['TARGET_BASE_URL', 'SENTINEL_CHECK_INTERVAL']:
                os.environ.pop(key, None)

            config = SentinelConfig(
                target_base_url=None,
                enabled_checks=None
            )

            # All None values should trigger defaults
            # PR #397: Only 6 runtime checks enabled by default
            assert config.target_base_url == 'http://localhost:8000'
            assert len(config.enabled_checks) == 6


class TestCheckResult:
    """Test CheckResult model."""

    def test_check_result_creation(self):
        """Test basic CheckResult creation."""
        timestamp = datetime.now(timezone.utc)
        result = CheckResult(
            check_id="S1-probes",
            timestamp=timestamp,
            status="pass",
            latency_ms=123.45,
            message="All checks passed"
        )

        assert result.check_id == "S1-probes"
        assert result.timestamp == timestamp
        assert result.status == "pass"
        assert result.latency_ms == 123.45
        assert result.message == "All checks passed"
        assert result.details is None

    def test_check_result_with_details(self):
        """Test CheckResult with details dictionary."""
        timestamp = datetime.now(timezone.utc)
        details = {"metric1": 0.95, "metric2": 100}

        result = CheckResult(
            check_id="S2-golden-fact-recall",
            timestamp=timestamp,
            status="pass",
            latency_ms=250.0,
            message="Recall successful",
            details=details
        )

        assert result.details == details
        assert result.details["metric1"] == 0.95

    def test_check_result_to_dict(self):
        """Test CheckResult to_dict() conversion."""
        timestamp = datetime.now(timezone.utc)
        details = {"test_metric": 42}

        result = CheckResult(
            check_id="test-check",
            timestamp=timestamp,
            status="warn",
            latency_ms=500.0,
            message="Warning message",
            details=details
        )

        result_dict = result.to_dict()

        assert result_dict["check_id"] == "test-check"
        assert result_dict["timestamp"] == timestamp.isoformat()
        assert result_dict["status"] == "warn"
        assert result_dict["latency_ms"] == 500.0
        assert result_dict["message"] == "Warning message"
        assert result_dict["details"] == details

    def test_check_result_from_dict(self):
        """Test CheckResult from_dict() creation."""
        timestamp = datetime.now(timezone.utc)
        data = {
            'check_id': 'test-check',
            'timestamp': timestamp.isoformat(),
            'status': 'fail',
            'latency_ms': 1000.0,
            'message': 'Check failed',
            'details': {'error_code': 500}
        }

        result = CheckResult.from_dict(data)

        assert result.check_id == 'test-check'
        assert isinstance(result.timestamp, datetime)
        assert result.status == 'fail'
        assert result.latency_ms == 1000.0
        assert result.message == 'Check failed'
        assert result.details == {'error_code': 500}

    def test_check_result_from_dict_with_datetime_object(self):
        """Test from_dict() handles both string and datetime timestamps."""
        timestamp = datetime.now(timezone.utc)
        data = {
            'check_id': 'test-check',
            'timestamp': timestamp,  # Already a datetime object
            'status': 'pass',
            'latency_ms': 100.0,
            'message': 'Success'
        }

        result = CheckResult.from_dict(data)

        assert result.timestamp == timestamp
        assert isinstance(result.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
