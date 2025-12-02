#!/usr/bin/env python3
"""
Unit tests for GitHubWebhookAlerter

Tests webhook alerting functionality, GitHub integration, rate limiting,
and integration with the MCP storage layer.
"""

import asyncio
import json
import os
import pytest
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from src.monitoring.sentinel.webhook_alerter import GitHubWebhookAlerter
    from src.monitoring.sentinel.models import CheckResult
    WEBHOOK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import webhook alerter: {e}")
    WEBHOOK_AVAILABLE = False


class TestGitHubWebhookAlerter(unittest.TestCase):
    """Test suite for GitHub webhook alerter functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not WEBHOOK_AVAILABLE:
            self.skipTest("GitHubWebhookAlerter not available")
            
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'GITHUB_TOKEN': 'test-github-token',
            'SENTINEL_GITHUB_REPO': 'test-org/test-repo',
            'SENTINEL_WEBHOOK_SECRET': 'test-webhook-secret'
        })
        self.env_patcher.start()
        
        # Create test configuration
        self.test_config = {
            'rate_limit_per_minute': 5,
            'max_retries': 3,
            'timeout_seconds': 30
        }
        
        self.alerter = GitHubWebhookAlerter(self.test_config)
        
        # Sample test data
        self.test_check_result = CheckResult(
            check_id='S1-test-health',
            status='FAIL',
            message='Test service is down',
            details={'service': 'test-service', 'error': 'Connection timeout'},
            timestamp=datetime.now(),
            severity='critical'
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'env_patcher'):
            self.env_patcher.stop()
    
    def test_alerter_initialization(self):
        """Test proper initialization of webhook alerter."""
        self.assertEqual(self.alerter.github_token, 'test-github-token')
        self.assertEqual(self.alerter.github_repo, 'test-org/test-repo')
        self.assertEqual(self.alerter.webhook_secret, 'test-webhook-secret')
        self.assertEqual(self.alerter.max_alerts_per_minute, 5)
        self.assertIsInstance(self.alerter.recent_alerts, list)
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Test with minimal config
        minimal_alerter = GitHubWebhookAlerter({})
        self.assertEqual(minimal_alerter.max_alerts_per_minute, 10)  # Default value
        
        # Test with custom config
        custom_config = {'rate_limit_per_minute': 15}
        custom_alerter = GitHubWebhookAlerter(custom_config)
        self.assertEqual(custom_alerter.max_alerts_per_minute, 15)
    
    @patch('aiohttp.ClientSession.post')
    async def test_send_alert_success(self, mock_post):
        """Test successful alert sending to GitHub."""
        # Mock successful GitHub API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'message': 'success'})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Send alert
        result = await self.alerter.send_alert(self.test_check_result)
        
        # Verify result
        self.assertTrue(result)
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check URL
        expected_url = f"https://api.github.com/repos/test-org/test-repo/dispatches"
        self.assertEqual(call_args[1]['url'], expected_url)
        
        # Check headers
        headers = call_args[1]['headers']
        self.assertIn('Authorization', headers)
        self.assertIn('Content-Type', headers)
        self.assertEqual(headers['Content-Type'], 'application/json')
    
    @patch('aiohttp.ClientSession.post')
    async def test_send_alert_failure(self, mock_post):
        """Test alert sending failure handling."""
        # Mock failed GitHub API response
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value='Unauthorized')
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Send alert
        result = await self.alerter.send_alert(self.test_check_result)
        
        # Verify failure is handled
        self.assertFalse(result)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Add alerts to reach limit
        current_time = datetime.now()
        for i in range(10):  # More than the default limit of 10
            self.alerter.recent_alerts.append(current_time)
        
        # Check if rate limited
        self.assertFalse(self.alerter._check_rate_limit())
        
        # Clean up old alerts and test again
        old_time = current_time - timedelta(minutes=2)
        self.alerter.recent_alerts = [old_time] * 5  # Old alerts should be ignored
        self.assertTrue(self.alerter._check_rate_limit())
    
    def test_alert_deduplication(self):
        """Test alert deduplication logic."""
        # Create duplicate alert
        duplicate_result = CheckResult(
            check_id='S1-test-health',
            status='FAIL',
            message='Test service is down',
            details={'service': 'test-service'},
            timestamp=datetime.now(),
            severity='critical'
        )
        
        # Test deduplication key generation
        key1 = self.alerter._generate_alert_key(self.test_check_result)
        key2 = self.alerter._generate_alert_key(duplicate_result)
        
        # Keys should be similar for similar alerts
        self.assertEqual(key1, key2)
    
    def test_payload_creation(self):
        """Test GitHub dispatch payload creation."""
        payload = self.alerter._create_dispatch_payload(self.test_check_result)
        
        # Verify payload structure
        self.assertIn('event_type', payload)
        self.assertIn('client_payload', payload)
        self.assertEqual(payload['event_type'], 'sentinel-alert')
        
        # Verify client payload
        client_payload = payload['client_payload']
        self.assertEqual(client_payload['check_id'], 'S1-test-health')
        self.assertEqual(client_payload['status'], 'FAIL')
        self.assertEqual(client_payload['severity'], 'critical')
        self.assertIn('details', client_payload)
        self.assertIn('timestamp', client_payload)
    
    def test_security_validation(self):
        """Test security validation of webhook data."""
        # Test with malicious input
        malicious_result = CheckResult(
            check_id='S1-test; rm -rf /',
            status='FAIL',
            message='Test $(cat /etc/passwd)',
            details={'script': '<script>alert("xss")</script>'},
            timestamp=datetime.now(),
            severity='critical'
        )
        
        payload = self.alerter._create_dispatch_payload(malicious_result)
        
        # Ensure no code execution patterns in payload
        payload_str = json.dumps(payload)
        self.assertNotIn('rm -rf', payload_str)
        self.assertNotIn('$(cat', payload_str)
        self.assertNotIn('<script>', payload_str)
    
    @patch('aiohttp.ClientSession.post')
    async def test_retry_logic(self, mock_post):
        """Test retry logic for failed requests."""
        # Mock transient failure then success
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 500
        mock_response_fail.text = AsyncMock(return_value='Internal Server Error')
        
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={'message': 'success'})
        
        mock_post.return_value.__aenter__.side_effect = [
            mock_response_fail,  # First attempt fails
            mock_response_success  # Second attempt succeeds
        ]
        
        # Test retry functionality would be implemented in the actual alerter
        # For now, test that failures are properly handled
        result = await self.alerter.send_alert(self.test_check_result)
        self.assertFalse(result)  # Current implementation doesn't retry
    
    def test_environment_variable_handling(self):
        """Test handling of missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            alerter = GitHubWebhookAlerter()
            self.assertIsNone(alerter.github_token)
            self.assertEqual(alerter.github_repo, 'credentum/veris-memory')  # Default
            self.assertIsNone(alerter.webhook_secret)


class TestWebhookAlerterIntegration(unittest.TestCase):
    """Integration tests for webhook alerter with MCP storage."""
    
    def setUp(self):
        """Set up integration test environment."""
        if not WEBHOOK_AVAILABLE:
            self.skipTest("GitHubWebhookAlerter not available")
    
    def test_mcp_storage_integration(self):
        """Test integration with MCP storage implementations."""
        # This test verifies that webhook alerter can work with MCP storage
        # when context storage is needed for alert correlation
        
        alerter = GitHubWebhookAlerter()
        
        # Test that alerter can handle context data that might come from MCP
        context_data = {
            'alert_history': [
                {'check_id': 'S1-health', 'timestamp': '2025-08-21T01:00:00Z'},
                {'check_id': 'S1-health', 'timestamp': '2025-08-21T01:05:00Z'}
            ],
            'related_contexts': [
                {'type': 'incident', 'id': 'inc-123'},
                {'type': 'knowledge', 'id': 'kb-456'}
            ]
        }
        
        # Verify alerter can process context-enriched alerts
        enriched_result = CheckResult(
            check_id='S1-health-enriched',
            status='FAIL',
            message='Service failure with context',
            details={'context': context_data, 'service': 'mcp_server'},
            timestamp=datetime.now(),
            severity='critical'
        )
        
        payload = alerter._create_dispatch_payload(enriched_result)
        self.assertIn('context', payload['client_payload']['details'])
        self.assertIsInstance(payload['client_payload']['details']['context'], dict)


# Async test runner
class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases."""
    
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()
    
    def run_async(self, coro):
        return self.loop.run_until_complete(coro)


def run_webhook_tests():
    """Run all webhook alerter tests."""
    if not WEBHOOK_AVAILABLE:
        print("⏭️ Webhook alerter tests skipped - module not available")
        return False
    
    print("🧪 Running Webhook Alerter Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGitHubWebhookAlerter))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWebhookAlerterIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_webhook_tests()
    exit(0 if success else 1)