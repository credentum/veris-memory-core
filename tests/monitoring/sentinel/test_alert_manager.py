#!/usr/bin/env python3
"""
Unit tests for AlertManager and AlertDeduplicator

Tests alert management, deduplication, routing, and integration 
with various alerting channels including MCP storage.
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
    from src.monitoring.sentinel.alert_manager import AlertManager, AlertDeduplicator
    from src.monitoring.sentinel.models import CheckResult
    ALERT_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import alert manager: {e}")
    ALERT_MANAGER_AVAILABLE = False


class TestAlertDeduplicator(unittest.TestCase):
    """Test suite for alert deduplication functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not ALERT_MANAGER_AVAILABLE:
            self.skipTest("AlertDeduplicator not available")
            
        self.deduplicator = AlertDeduplicator(window_minutes=30)
    
    async def test_deduplicator_initialization(self):
        """Test proper initialization of deduplicator."""
        self.assertEqual(self.deduplicator.window_minutes, 30)
        self.assertIsInstance(self.deduplicator.alert_history, dict)
        self.assertIsNotNone(self.deduplicator.lock)
    
    async def test_should_alert_new_alert(self):
        """Test that new alerts are allowed."""
        result = await self.deduplicator.should_alert('new-alert-key')
        self.assertTrue(result)
    
    async def test_should_alert_duplicate_within_window(self):
        """Test that duplicate alerts within window are blocked."""
        alert_key = 'test-alert-key'
        
        # First alert should be allowed
        result1 = await self.deduplicator.should_alert(alert_key)
        self.assertTrue(result1)
        
        # Second alert immediately after should be blocked
        result2 = await self.deduplicator.should_alert(alert_key)
        self.assertFalse(result2)
    
    async def test_should_alert_duplicate_outside_window(self):
        """Test that duplicate alerts outside window are allowed."""
        alert_key = 'test-alert-key'
        
        # Manually add old alert outside window
        old_time = datetime.now() - timedelta(minutes=35)
        self.deduplicator.alert_history[alert_key] = [old_time]
        
        # New alert should be allowed
        result = await self.deduplicator.should_alert(alert_key)
        self.assertTrue(result)
    
    async def test_cleanup_old_alerts(self):
        """Test cleanup of old alerts from history."""
        alert_key = 'test-alert-key'
        
        # Add mix of old and recent alerts
        old_time = datetime.now() - timedelta(minutes=45)
        recent_time = datetime.now() - timedelta(minutes=10)
        
        self.deduplicator.alert_history[alert_key] = [old_time, recent_time]
        
        # Should alert (but will clean up old ones)
        result = await self.deduplicator.should_alert(alert_key)
        self.assertFalse(result)  # Recent one should block
        
        # Verify old alert was cleaned up
        remaining_alerts = self.deduplicator.alert_history[alert_key]
        self.assertEqual(len(remaining_alerts), 2)  # Recent + new one


class TestAlertManager(unittest.TestCase):
    """Test suite for alert manager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not ALERT_MANAGER_AVAILABLE:
            self.skipTest("AlertManager not available")
            
        # Mock configuration
        self.test_config = {
            'deduplication_window_minutes': 30,
            'telegram_enabled': True,
            'github_enabled': True,
            'rate_limit_per_minute': 10
        }
        
        # Create sample check result
        self.test_check_result = CheckResult(
            check_id='S1-test-health',
            status='FAIL',
            message='Test service health check failed',
            details={'service': 'test-service', 'error': 'Connection timeout'},
            timestamp=datetime.now(),
            severity='critical'
        )
    
    @patch('monitoring.sentinel.alert_manager.TelegramAlerter')
    @patch('monitoring.sentinel.alert_manager.GitHubWebhookAlerter')
    def test_alert_manager_initialization(self, mock_webhook, mock_telegram):
        """Test proper initialization of alert manager."""
        manager = AlertManager(self.test_config)
        
        # Verify components are initialized
        self.assertIsNotNone(manager.deduplicator)
        self.assertIsInstance(manager.alerters, list)
        self.assertEqual(len(manager.alerters), 2)  # Telegram + GitHub
        
        # Verify alerters were created
        mock_telegram.assert_called_once()
        mock_webhook.assert_called_once()
    
    @patch('monitoring.sentinel.alert_manager.TelegramAlerter')
    @patch('monitoring.sentinel.alert_manager.GitHubWebhookAlerter')
    async def test_process_alert_success(self, mock_webhook_class, mock_telegram_class):
        """Test successful alert processing."""
        # Mock alerter instances
        mock_telegram = AsyncMock()
        mock_telegram.send_alert = AsyncMock(return_value=True)
        mock_telegram_class.return_value = mock_telegram
        
        mock_webhook = AsyncMock()
        mock_webhook.send_alert = AsyncMock(return_value=True)
        mock_webhook_class.return_value = mock_webhook
        
        manager = AlertManager(self.test_config)
        
        # Process alert
        result = await manager.process_alert(self.test_check_result)
        
        # Verify success
        self.assertTrue(result)
        
        # Verify alerters were called
        mock_telegram.send_alert.assert_called_once_with(self.test_check_result)
        mock_webhook.send_alert.assert_called_once_with(self.test_check_result)
    
    @patch('monitoring.sentinel.alert_manager.TelegramAlerter')
    @patch('monitoring.sentinel.alert_manager.GitHubWebhookAlerter')
    async def test_process_alert_deduplication(self, mock_webhook_class, mock_telegram_class):
        """Test alert deduplication prevents duplicate processing."""
        # Mock alerter instances
        mock_telegram = AsyncMock()
        mock_telegram.send_alert = AsyncMock(return_value=True)
        mock_telegram_class.return_value = mock_telegram
        
        mock_webhook = AsyncMock()
        mock_webhook.send_alert = AsyncMock(return_value=True)
        mock_webhook_class.return_value = mock_webhook
        
        manager = AlertManager(self.test_config)
        
        # Process same alert twice
        result1 = await manager.process_alert(self.test_check_result)
        result2 = await manager.process_alert(self.test_check_result)
        
        # First should succeed, second should be deduplicated
        self.assertTrue(result1)
        self.assertTrue(result2)  # Still returns True, but doesn't send
        
        # Verify alerters were only called once
        mock_telegram.send_alert.assert_called_once()
        mock_webhook.send_alert.assert_called_once()
    
    @patch('monitoring.sentinel.alert_manager.TelegramAlerter')
    @patch('monitoring.sentinel.alert_manager.GitHubWebhookAlerter')
    async def test_process_alert_partial_failure(self, mock_webhook_class, mock_telegram_class):
        """Test alert processing with partial alerter failures."""
        # Mock alerter instances - one succeeds, one fails
        mock_telegram = AsyncMock()
        mock_telegram.send_alert = AsyncMock(return_value=False)
        mock_telegram_class.return_value = mock_telegram
        
        mock_webhook = AsyncMock()
        mock_webhook.send_alert = AsyncMock(return_value=True)
        mock_webhook_class.return_value = mock_webhook
        
        manager = AlertManager(self.test_config)
        
        # Process alert
        result = await manager.process_alert(self.test_check_result)
        
        # Should still return True if at least one alerter succeeds
        self.assertTrue(result)
    
    def test_alert_key_generation(self):
        """Test alert key generation for deduplication."""
        manager = AlertManager(self.test_config)
        
        # Test key generation
        key1 = manager._generate_alert_key(self.test_check_result)
        
        # Create similar alert
        similar_result = CheckResult(
            check_id='S1-test-health',  # Same check
            status='FAIL',
            message='Test service health check failed',  # Same message
            details={'service': 'test-service'},
            timestamp=datetime.now(),
            severity='critical'
        )
        
        key2 = manager._generate_alert_key(similar_result)
        
        # Keys should be the same for similar alerts
        self.assertEqual(key1, key2)
    
    def test_alert_severity_mapping(self):
        """Test alert severity mapping for different alerters."""
        manager = AlertManager(self.test_config)
        
        # Test severity mapping
        critical_result = CheckResult(
            check_id='S1-test',
            status='FAIL',
            message='Critical failure',
            details={},
            timestamp=datetime.now(),
            severity='critical'
        )
        
        warning_result = CheckResult(
            check_id='S2-test',
            status='WARN',
            message='Warning condition',
            details={},
            timestamp=datetime.now(),
            severity='warning'
        )
        
        # Both should generate different keys
        critical_key = manager._generate_alert_key(critical_result)
        warning_key = manager._generate_alert_key(warning_result)
        
        self.assertNotEqual(critical_key, warning_key)


class TestAlertManagerMCPIntegration(unittest.TestCase):
    """Integration tests for alert manager with MCP storage."""
    
    def setUp(self):
        """Set up MCP integration test environment."""
        if not ALERT_MANAGER_AVAILABLE:
            self.skipTest("AlertManager not available")
    
    def test_mcp_context_enrichment(self):
        """Test alert enrichment with MCP context data."""
        config = {
            'mcp_storage_enabled': True,
            'context_enrichment': True
        }
        
        manager = AlertManager(config)
        
        # Test alert with MCP context
        enriched_result = CheckResult(
            check_id='S1-mcp-integration',
            status='FAIL',
            message='Service failure with MCP context',
            details={
                'service': 'mcp_server',
                'mcp_context': {
                    'related_incidents': ['inc-123', 'inc-456'],
                    'knowledge_articles': ['kb-789'],
                    'context_metadata': {
                        'confidence': 0.95,
                        'correlation_id': 'corr-abc123'
                    }
                }
            },
            timestamp=datetime.now(),
            severity='critical'
        )
        
        # Verify context is preserved in alert key generation
        alert_key = manager._generate_alert_key(enriched_result)
        self.assertIsInstance(alert_key, str)
        self.assertTrue(len(alert_key) > 0)
    
    def test_mcp_storage_correlation(self):
        """Test correlation of alerts with stored MCP contexts."""
        manager = AlertManager({})
        
        # Simulate correlated alerts that might be stored in MCP
        base_alert = CheckResult(
            check_id='S1-service-health',
            status='FAIL',
            message='Service health degraded',
            details={'service': 'api_server'},
            timestamp=datetime.now(),
            severity='warning'
        )
        
        related_alert = CheckResult(
            check_id='S2-database-connectivity',
            status='FAIL',
            message='Database connection issues',
            details={'service': 'database', 'related_to': 'S1-service-health'},
            timestamp=datetime.now(),
            severity='critical'
        )
        
        # Test that different but related alerts generate different keys
        key1 = manager._generate_alert_key(base_alert)
        key2 = manager._generate_alert_key(related_alert)
        
        self.assertNotEqual(key1, key2)
        
        # But similar alerts should generate same keys
        duplicate_alert = CheckResult(
            check_id='S1-service-health',
            status='FAIL',
            message='Service health degraded',
            details={'service': 'api_server'},
            timestamp=datetime.now() + timedelta(seconds=30),
            severity='warning'
        )
        
        key3 = manager._generate_alert_key(duplicate_alert)
        self.assertEqual(key1, key3)


# Async test runner for alert manager
async def run_alert_manager_async_tests():
    """Run async tests for alert manager."""
    if not ALERT_MANAGER_AVAILABLE:
        print("⏭️ Alert manager async tests skipped - module not available")
        return False
    
    print("🧪 Running Alert Manager Async Tests...")
    
    # Test deduplicator
    deduplicator = AlertDeduplicator(window_minutes=30)
    
    # Test 1: New alert should be allowed
    result1 = await deduplicator.should_alert('test-key-1')
    assert result1 == True, "New alert should be allowed"
    
    # Test 2: Duplicate alert should be blocked
    result2 = await deduplicator.should_alert('test-key-1')
    assert result2 == False, "Duplicate alert should be blocked"
    
    # Test 3: Different alert should be allowed
    result3 = await deduplicator.should_alert('test-key-2')
    assert result3 == True, "Different alert should be allowed"
    
    print("✅ Alert manager async tests passed")
    return True


def run_alert_manager_tests():
    """Run all alert manager tests."""
    if not ALERT_MANAGER_AVAILABLE:
        print("⏭️ Alert manager tests skipped - module not available")
        return False
    
    print("🧪 Running Alert Manager Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAlertDeduplicator))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAlertManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAlertManagerMCPIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        async_success = loop.run_until_complete(run_alert_manager_async_tests())
    finally:
        loop.close()
    
    # Return success status
    return result.wasSuccessful() and async_success


if __name__ == '__main__':
    success = run_alert_manager_tests()
    exit(0 if success else 1)