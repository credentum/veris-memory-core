#!/usr/bin/env python3
"""
Unit tests for Alert Manager.

Tests the AlertManager class with mocked dependencies.
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

from src.monitoring.sentinel.alert_manager import (
    AlertManager, AlertDeduplicator, GitHubIssueCreator
)
from src.monitoring.sentinel.telegram_alerter import AlertSeverity
from src.monitoring.sentinel.models import CheckResult


class TestAlertDeduplicator:
    """Test suite for AlertDeduplicator."""
    
    @pytest.fixture
    def deduplicator(self):
        """Create a deduplicator instance."""
        return AlertDeduplicator(window_minutes=30)
    
    def test_initialization(self):
        """Test deduplicator initialization."""
        dedup = AlertDeduplicator(window_minutes=60)
        assert dedup.window_minutes == 60
        assert len(dedup.alert_history) == 0
    
    @pytest.mark.asyncio
    async def test_should_alert_first_time(self, deduplicator):
        """Test that first alert should be sent."""
        alert_key = "test_alert_1"
        should_send = await deduplicator.should_alert(alert_key)
        assert should_send is True
        assert len(deduplicator.alert_history[alert_key]) == 1
    
    @pytest.mark.asyncio
    async def test_should_alert_duplicate(self, deduplicator):
        """Test that duplicate alerts are suppressed."""
        alert_key = "test_alert_2"
        
        # First alert should be sent
        assert await deduplicator.should_alert(alert_key) is True
        
        # Duplicate should be suppressed
        assert await deduplicator.should_alert(alert_key) is False
    
    @pytest.mark.asyncio
    async def test_should_alert_after_window(self, deduplicator):
        """Test that alerts are allowed after window expires."""
        deduplicator.window_minutes = 0.01  # Very short window for testing
        alert_key = "test_alert_3"
        
        # First alert
        assert await deduplicator.should_alert(alert_key) is True
        
        # Wait for window to expire
        await asyncio.sleep(0.02)
        
        # Should allow new alert
        assert await deduplicator.should_alert(alert_key) is True
    
    def test_get_alert_key(self, deduplicator):
        """Test alert key generation."""
        key1 = deduplicator.get_alert_key("check1", "fail", "Error message")
        key2 = deduplicator.get_alert_key("check1", "fail", "Error message")
        key3 = deduplicator.get_alert_key("check2", "fail", "Error message")
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different inputs should generate different key
        assert key1 != key3
    
    def test_get_alert_key_truncation(self, deduplicator):
        """Test that long messages are truncated in key generation."""
        long_message = "x" * 200
        key1 = deduplicator.get_alert_key("check", "fail", long_message)
        key2 = deduplicator.get_alert_key("check", "fail", long_message[:100])
        
        # Should use only first 100 chars
        assert key1 == key2
    
    @pytest.mark.asyncio
    async def test_cleanup_old_entries(self, deduplicator):
        """Test that old entries are cleaned up."""
        alert_key = "test_cleanup"
        
        # Add old entry manually
        old_time = datetime.utcnow() - timedelta(minutes=60)
        deduplicator.alert_history[alert_key] = [old_time]
        
        # Add recent entry
        await deduplicator.should_alert(alert_key)
        
        # Old entry should be removed, new one added
        assert len(deduplicator.alert_history[alert_key]) == 1
        assert deduplicator.alert_history[alert_key][0] > old_time


class TestGitHubIssueCreator:
    """Test suite for GitHubIssueCreator."""
    
    @pytest.fixture
    def creator(self):
        """Create a GitHub issue creator instance."""
        return GitHubIssueCreator(
            token="test_token",
            repo="owner/repo",
            labels=["test", "automated"]
        )
    
    def test_initialization(self):
        """Test GitHub issue creator initialization."""
        creator = GitHubIssueCreator("token", "owner/repo", ["label1"])
        
        assert creator.token == "token"
        assert creator.repo == "owner/repo"
        assert creator.labels == ["label1"]
        assert creator.api_url == "https://api.github.com/repos/owner/repo/issues"
        assert creator.headers["Authorization"] == "token token"
    
    @pytest.mark.asyncio
    async def test_create_issue_success(self, creator):
        """Test successful issue creation."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json = AsyncMock(return_value={
                "html_url": "https://github.com/owner/repo/issues/123"
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            url = await creator.create_issue(
                title="Test Issue",
                body="Test body",
                labels=["bug"]
            )
            
            assert url == "https://github.com/owner/repo/issues/123"
    
    @pytest.mark.asyncio
    async def test_create_issue_failure(self, creator):
        """Test issue creation failure."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 403
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            url = await creator.create_issue("Test", "Body")
            
            assert url is None
    
    @pytest.mark.asyncio
    async def test_create_issue_exception(self, creator):
        """Test issue creation with exception."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.side_effect = Exception("Network error")
            
            url = await creator.create_issue("Test", "Body")
            
            assert url is None


class TestAlertManager:
    """Test suite for AlertManager."""
    
    @pytest.fixture
    def manager(self):
        """Create an AlertManager instance."""
        with patch('src.monitoring.sentinel.alert_manager.TelegramAlerter'):
            with patch('src.monitoring.sentinel.alert_manager.GitHubIssueCreator'):
                return AlertManager(
                    telegram_token="FAKE_TG_TOKEN_123456789:TEST_HASH_ONLY",
                    telegram_chat_id="FAKE_CHAT_123456",
                    github_token="FAKE_GH_TOKEN_ghp_TEST123456789",
                    github_repo="test-owner/test-repo",
                    dedup_window_minutes=30,
                    alert_threshold_failures=3
                )
    
    def test_initialization_with_telegram(self):
        """Test initialization with Telegram configured."""
        manager = AlertManager(
            telegram_token="token",
            telegram_chat_id="chat"
        )
        
        assert manager.telegram is not None
        assert manager.github is None
        assert manager.alert_threshold_failures == 3
    
    def test_initialization_with_github(self):
        """Test initialization with GitHub configured."""
        manager = AlertManager(
            github_token="token",
            github_repo="owner/repo"
        )
        
        assert manager.telegram is None
        assert manager.github is not None
    
    def test_initialization_with_both(self):
        """Test initialization with both services configured."""
        manager = AlertManager(
            telegram_token="tg_token",
            telegram_chat_id="chat",
            github_token="gh_token",
            github_repo="owner/repo"
        )
        
        assert manager.telegram is not None
        assert manager.github is not None
    
    def test_determine_severity_pass(self, manager):
        """Test severity determination for passing checks."""
        result = CheckResult(
            check_id="test",
            timestamp=datetime.utcnow(),
            status="pass",
            latency_ms=100.0,
            message="OK"
        )
        
        severity = manager._determine_severity(result)
        assert severity == AlertSeverity.INFO
    
    def test_determine_severity_critical_checks(self, manager):
        """Test severity for critical checks."""
        critical_checks = ["S1-health-probes", "S5-security-negatives", "S6-backup-restore"]
        
        for check_id in critical_checks:
            result = CheckResult(
                check_id=check_id,
                timestamp=datetime.utcnow(),
                status="fail",
                latency_ms=100.0,
                message="Failed"
            )
            
            severity = manager._determine_severity(result)
            assert severity == AlertSeverity.CRITICAL
    
    def test_determine_severity_high_checks(self, manager):
        """Test severity for high priority checks."""
        high_checks = ["S2-golden-fact-recall", "S8-capacity-smoke", "S4-metrics-wiring"]
        
        for check_id in high_checks:
            result = CheckResult(
                check_id=check_id,
                timestamp=datetime.utcnow(),
                status="fail",
                latency_ms=100.0,
                message="Failed"
            )
            
            severity = manager._determine_severity(result)
            assert severity == AlertSeverity.HIGH
    
    def test_determine_severity_other_checks(self, manager):
        """Test severity for other checks."""
        result = CheckResult(
            check_id="S99-unknown-check",
            timestamp=datetime.utcnow(),
            status="fail",
            latency_ms=100.0,
            message="Failed"
        )
        
        severity = manager._determine_severity(result)
        assert severity == AlertSeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_update_failure_tracking(self, manager):
        """Test failure count tracking."""
        result_fail = CheckResult(
            check_id="test-check",
            timestamp=datetime.utcnow(),
            status="fail",
            latency_ms=100.0,
            message="Failed"
        )
        
        result_pass = CheckResult(
            check_id="test-check",
            timestamp=datetime.utcnow(),
            status="pass",
            latency_ms=100.0,
            message="OK"
        )
        
        # Track failures
        await manager._update_failure_tracking(result_fail)
        assert manager.failure_counts["test-check"] == 1
        
        await manager._update_failure_tracking(result_fail)
        assert manager.failure_counts["test-check"] == 2
        
        # Reset on success
        await manager._update_failure_tracking(result_pass)
        assert manager.failure_counts["test-check"] == 0
    
    @pytest.mark.asyncio
    async def test_should_alert_info_level(self, manager):
        """Test that info level alerts are skipped."""
        result = CheckResult(
            check_id="test",
            timestamp=datetime.utcnow(),
            status="pass",
            latency_ms=100.0,
            message="OK"
        )
        
        should_alert = await manager._should_alert(result, AlertSeverity.INFO)
        assert should_alert is False
    
    @pytest.mark.asyncio
    async def test_should_alert_threshold(self, manager):
        """Test alert threshold checking."""
        result = CheckResult(
            check_id="test",
            timestamp=datetime.utcnow(),
            status="fail",
            latency_ms=100.0,
            message="Failed"
        )
        
        # First two failures should not alert
        manager.failure_counts["test"] = 1
        should_alert = await manager._should_alert(result, AlertSeverity.HIGH)
        assert should_alert is False
        
        manager.failure_counts["test"] = 2
        should_alert = await manager._should_alert(result, AlertSeverity.HIGH)
        assert should_alert is False
        
        # Third failure should alert
        manager.failure_counts["test"] = 3
        with patch.object(manager.deduplicator, 'should_alert', return_value=True):
            should_alert = await manager._should_alert(result, AlertSeverity.HIGH)
            assert should_alert is True
    
    @pytest.mark.asyncio
    async def test_route_alert_critical(self, manager):
        """Test routing for critical alerts."""
        result = CheckResult(
            check_id="S5-security-negatives",
            timestamp=datetime.utcnow(),
            status="fail",
            latency_ms=100.0,
            message="Security breach"
        )
        
        with patch.object(manager, '_send_telegram_alert') as mock_telegram:
            with patch.object(manager, '_create_github_issue') as mock_github:
                await manager._route_alert(result, AlertSeverity.CRITICAL)
                
                mock_telegram.assert_called_once()
                mock_github.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_route_alert_high(self, manager):
        """Test routing for high priority alerts."""
        result = CheckResult(
            check_id="test",
            timestamp=datetime.utcnow(),
            status="fail",
            latency_ms=100.0,
            message="Error"
        )
        
        with patch.object(manager, '_send_telegram_alert') as mock_telegram:
            with patch.object(manager, '_create_github_issue') as mock_github:
                await manager._route_alert(result, AlertSeverity.HIGH)
                
                mock_telegram.assert_called_once()
                mock_github.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_check_result_full_flow(self, manager):
        """Test complete flow of processing a check result."""
        result = CheckResult(
            check_id="S5-security-negatives",
            timestamp=datetime.utcnow(),
            status="fail",
            latency_ms=100.0,
            message="Security issue"
        )
        
        # Set up to trigger alert
        manager.failure_counts["S5-security-negatives"] = 2
        
        with patch.object(manager.deduplicator, 'should_alert', return_value=True):
            with patch.object(manager, '_route_alert') as mock_route:
                await manager.process_check_result(result)
                
                # Should have incremented failure count
                assert manager.failure_counts["S5-security-negatives"] == 3
                
                # Should have routed alert
                mock_route.assert_called_once()
                call_args = mock_route.call_args[0]
                assert call_args[0] == result
                assert call_args[1] == AlertSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_send_summary(self, manager):
        """Test sending summary report."""
        check_results = [
            CheckResult(
                check_id=f"test-{i}",
                timestamp=datetime.utcnow(),
                status="pass" if i < 8 else "fail",
                latency_ms=50.0 + i,
                message="Test"
            )
            for i in range(10)
        ]
        
        with patch.object(manager.telegram, 'send_summary', return_value=True) as mock_send:
            await manager.send_summary(24, check_results)
            
            mock_send.assert_called_once()
            call_kwargs = mock_send.call_args[1]
            
            assert call_kwargs['period_hours'] == 24
            assert call_kwargs['total_checks'] == 10
            assert call_kwargs['passed_checks'] == 8
            assert call_kwargs['failed_checks'] == 2
            assert call_kwargs['uptime_percent'] == 80.0
    
    @pytest.mark.asyncio
    async def test_send_summary_no_telegram(self):
        """Test summary with no Telegram configured."""
        manager = AlertManager()  # No services configured
        
        # Should not raise error
        await manager.send_summary(24, [])
    
    @pytest.mark.asyncio
    async def test_test_alerting(self, manager):
        """Test alerting channel testing."""
        with patch.object(manager.telegram, 'test_connection', return_value=True):
            with patch('aiohttp.ClientSession') as mock_session:
                mock_response = AsyncMock()
                mock_response.status = 200
                
                mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
                
                results = await manager.test_alerting()
                
                assert results['telegram'] is True
                assert results['github'] is True
    
    @pytest.mark.asyncio
    async def test_send_telegram_alert(self, manager):
        """Test sending Telegram alert."""
        result = CheckResult(
            check_id="test",
            timestamp=datetime.utcnow(),
            status="fail",
            latency_ms=100.0,
            message="Error",
            details={"key": "value"}
        )
        
        with patch.object(manager.telegram, 'send_alert', return_value=True) as mock_send:
            await manager._send_telegram_alert(result, AlertSeverity.HIGH)
            
            mock_send.assert_called_once()
            call_kwargs = mock_send.call_args[1]
            
            assert call_kwargs['check_id'] == "test"
            assert call_kwargs['status'] == "fail"
            assert call_kwargs['severity'] == AlertSeverity.HIGH
            assert call_kwargs['details'] == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_create_github_issue(self, manager):
        """Test GitHub issue creation."""
        result = CheckResult(
            check_id="S5-security",
            timestamp=datetime.utcnow(),
            status="fail",
            latency_ms=100.0,
            message="Security breach",
            details={"attempts": 10}
        )
        
        with patch.object(manager.github, 'create_issue', return_value="https://github.com/owner/repo/issues/1") as mock_create:
            with patch.object(manager.telegram, 'send_alert') as mock_telegram:
                await manager._create_github_issue(result)
                
                mock_create.assert_called_once()
                call_args = mock_create.call_args[0]
                
                assert "[Sentinel] Critical: S5-security failure" in call_args[0]
                assert "Security breach" in call_args[1]
                assert "attempts" in call_args[1]
                
                # Should send follow-up Telegram
                mock_telegram.assert_called_once()