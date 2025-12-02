#!/usr/bin/env python3
"""
Unit tests for SimpleRedisClient

Tests password extraction, connection logic, and regex parsing edge cases
for the deployment-critical SimpleRedisClient component.
"""

import os
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import redis

from src.storage.simple_redis import SimpleRedisClient


class TestSimpleRedisClientPasswordParsing(unittest.TestCase):
    """Test password extraction from redis:// URLs."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = SimpleRedisClient()

    def tearDown(self):
        """Clean up after tests."""
        if self.client:
            self.client.close()

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'redis://:testpassword@testhost:6380/2'
    }, clear=False)
    def test_password_extraction_from_url(self, mock_redis_class):
        """Test password is correctly extracted from redis://:password@host:port/db URL."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        # Verify Redis was called with correct parameters including password
        call_kwargs = mock_redis_class.call_args[1]
        self.assertEqual(call_kwargs['host'], 'testhost')
        self.assertEqual(call_kwargs['port'], 6380)
        self.assertEqual(call_kwargs['db'], 2)
        self.assertEqual(call_kwargs['password'], 'testpassword')

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'redis://testhost:6379/1'
    }, clear=False)
    def test_url_without_password(self, mock_redis_class):
        """Test URL parsing when no password is present."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        call_kwargs = mock_redis_class.call_args[1]
        self.assertEqual(call_kwargs['host'], 'testhost')
        self.assertEqual(call_kwargs['port'], 6379)
        self.assertEqual(call_kwargs['db'], 1)
        self.assertNotIn('password', call_kwargs)

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'redis://:envpassword@envhost:6379/0',
        'REDIS_PASSWORD': 'fallbackpassword'
    }, clear=False)
    def test_password_priority_url_over_env(self, mock_redis_class):
        """Test password from URL takes priority over REDIS_PASSWORD env var."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        call_kwargs = mock_redis_class.call_args[1]
        self.assertEqual(call_kwargs['password'], 'envpassword')  # URL password, not env var

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'redis://host:6379/0',
        'REDIS_PASSWORD': 'fallbackpassword'
    }, clear=False)
    def test_password_priority_env_var_fallback(self, mock_redis_class):
        """Test REDIS_PASSWORD env var is used when URL has no password."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        call_kwargs = mock_redis_class.call_args[1]
        self.assertEqual(call_kwargs['password'], 'fallbackpassword')

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'redis://:urlpassword@host:6379/0'
    }, clear=False)
    def test_password_priority_parameter_over_url(self, mock_redis_class):
        """Test password parameter takes priority over URL password."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect(redis_password='parampassword')

        self.assertTrue(result)
        call_kwargs = mock_redis_class.call_args[1]
        self.assertEqual(call_kwargs['password'], 'parampassword')  # Parameter wins


class TestSimpleRedisClientRegexEdgeCases(unittest.TestCase):
    """Test regex parsing edge cases and malformed URLs."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = SimpleRedisClient()

    def tearDown(self):
        """Clean up after tests."""
        if self.client:
            self.client.close()

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'not-a-valid-redis-url',
        'REDIS_HOST': 'fallbackhost',
        'REDIS_PORT': '7777',
        'REDIS_DB': '3'
    }, clear=False)
    @patch('src.storage.simple_redis.logger')
    def test_malformed_url_fallback_to_env_vars(self, mock_logger, mock_redis_class):
        """Test malformed URL triggers fallback to individual env vars."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        self.assertIn('Failed to parse REDIS_URL', str(mock_logger.warning.call_args))

        # Verify fallback to env vars
        call_kwargs = mock_redis_class.call_args[1]
        self.assertEqual(call_kwargs['host'], 'fallbackhost')
        self.assertEqual(call_kwargs['port'], 7777)
        self.assertEqual(call_kwargs['db'], 3)

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'redis://host'  # No port, no db
    }, clear=False)
    def test_url_with_defaults(self, mock_redis_class):
        """Test URL parsing with missing optional components (port, db)."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        call_kwargs = mock_redis_class.call_args[1]
        self.assertEqual(call_kwargs['host'], 'host')
        self.assertEqual(call_kwargs['port'], 6379)  # Default port
        self.assertEqual(call_kwargs['db'], 0)  # Default db

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'redis://:@host:6379/0'  # Empty password
    }, clear=False)
    def test_url_with_empty_password(self, mock_redis_class):
        """Test URL with empty password string."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        call_kwargs = mock_redis_class.call_args[1]
        # Empty password should be treated as no password
        self.assertNotIn('password', call_kwargs)

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'redis://:pass@word:with:colons@host:6379/0'
    }, clear=False)
    def test_url_with_password_containing_special_chars(self, mock_redis_class):
        """Test URL password extraction with special characters."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        call_kwargs = mock_redis_class.call_args[1]
        # Password should include colons and special chars
        self.assertEqual(call_kwargs['password'], 'pass@word:with:colons')


class TestSimpleRedisClientConnectionBehavior(unittest.TestCase):
    """Test connection behavior with and without authentication."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = SimpleRedisClient()

    def tearDown(self):
        """Clean up after tests."""
        if self.client:
            self.client.close()

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    def test_connection_without_authentication(self, mock_redis_class):
        """Test connection succeeds without authentication (default behavior)."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        self.assertTrue(self.client.is_connected)
        call_kwargs = mock_redis_class.call_args[1]
        self.assertNotIn('password', call_kwargs)

    @patch('redis.Redis')
    @patch.dict(os.environ, {
        'REDIS_URL': 'redis://:authpassword@host:6379/0'
    }, clear=False)
    def test_connection_with_authentication(self, mock_redis_class):
        """Test connection succeeds with password authentication."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        self.assertTrue(self.client.is_connected)
        call_kwargs = mock_redis_class.call_args[1]
        self.assertEqual(call_kwargs['password'], 'authpassword')

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    @patch('src.storage.simple_redis.logger')
    def test_connection_failure_handling(self, mock_logger, mock_redis_class):
        """Test connection failure is handled gracefully."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.side_effect = redis.ConnectionError("Connection refused")
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertFalse(result)
        self.assertFalse(self.client.is_connected)
        self.assertIsNone(self.client.client)
        mock_logger.error.assert_called_once()

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    @patch('src.storage.simple_redis.logger')
    def test_debug_logging_on_success(self, mock_logger, mock_redis_class):
        """Test debug logging includes connection parameters on success."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        result = self.client.connect()

        self.assertTrue(result)
        # Verify debug logging was called with connection details
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        # Should log connection details with host, port, db
        self.assertTrue(any('db=' in call for call in debug_calls))


class TestSimpleRedisClientOperations(unittest.TestCase):
    """Test basic Redis operations (set, get, exists, delete)."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = SimpleRedisClient()

    def tearDown(self):
        """Clean up after tests."""
        if self.client:
            self.client.close()

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    def test_set_operation(self, mock_redis_class):
        """Test SET operation."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.set.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        self.client.connect()
        result = self.client.set('key1', 'value1')

        self.assertTrue(result)
        mock_redis_instance.set.assert_called_once_with('key1', 'value1')

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    def test_get_operation(self, mock_redis_class):
        """Test GET operation."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = 'retrieved_value'
        mock_redis_class.return_value = mock_redis_instance

        self.client.connect()
        result = self.client.get('key1')

        self.assertEqual(result, 'retrieved_value')
        mock_redis_instance.get.assert_called_once_with('key1')

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    def test_reconnection_on_connection_error(self, mock_redis_class):
        """Test automatic reconnection attempt when connection is lost."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.set.side_effect = redis.ConnectionError("Connection lost")
        mock_redis_class.return_value = mock_redis_instance

        self.client.connect()
        result = self.client.set('key1', 'value1')

        self.assertFalse(result)
        self.assertFalse(self.client.is_connected)


class TestSimpleRedisClientSetex(unittest.TestCase):
    """Test SETEX operation (set with expiration)."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = SimpleRedisClient()

    def tearDown(self):
        """Clean up after tests."""
        if self.client:
            self.client.close()

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    def test_setex_operation(self, mock_redis_class):
        """Test SETEX operation with TTL."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        self.client.connect()
        result = self.client.setex('key1', 300, 'value1')

        self.assertTrue(result)
        mock_redis_instance.setex.assert_called_once_with('key1', 300, 'value1')

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    def test_setex_with_json_value(self, mock_redis_class):
        """Test SETEX with JSON-serialized value (common use case)."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        import json
        json_value = json.dumps({'query': 'test', 'results': [1, 2, 3]})
        self.client.connect()
        result = self.client.setex('cache:query:hash', 600, json_value)

        self.assertTrue(result)
        mock_redis_instance.setex.assert_called_once_with('cache:query:hash', 600, json_value)

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    def test_setex_connection_error(self, mock_redis_class):
        """Test SETEX handles connection errors gracefully."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.side_effect = redis.ConnectionError("Connection lost")
        mock_redis_class.return_value = mock_redis_instance

        self.client.connect()
        result = self.client.setex('key1', 300, 'value1')

        self.assertFalse(result)
        self.assertFalse(self.client.is_connected)

    @patch('redis.Redis')
    @patch.dict(os.environ, {}, clear=True)
    def test_setex_auto_reconnect(self, mock_redis_class):
        """Test SETEX attempts reconnection when not connected."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        # Don't connect initially
        self.client.is_connected = False

        result = self.client.setex('key1', 300, 'value1')

        # Should have attempted reconnection
        self.assertTrue(result)
        mock_redis_instance.ping.assert_called()  # Reconnection attempt
        mock_redis_instance.setex.assert_called_once_with('key1', 300, 'value1')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
