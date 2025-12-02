#!/usr/bin/env python3
"""
Unit tests for Telegram Alerter.

Tests the TelegramAlerter class with mocked API calls.
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
import aiohttp

from src.monitoring.sentinel.telegram_alerter import (
    TelegramAlerter, AlertSeverity, TelegramMessage
)


class TestTelegramAlerter:
    """Test suite for TelegramAlerter."""
    
    @pytest.fixture
    def alerter(self):
        """Create a TelegramAlerter instance."""
        # Mock the validation methods for testing
        with patch.object(TelegramAlerter, '_validate_bot_token', return_value=True):
            with patch.object(TelegramAlerter, '_validate_chat_id', return_value=True):
                return TelegramAlerter(
                    bot_token="TEST_TOKEN_123456789:FAKE_HASH_FOR_TESTING_ONLY",
                    chat_id="TEST_CHAT_12345",
                    rate_limit=30
                )
    
    def test_initialization(self):
        """Test alerter initialization."""
        with patch.object(TelegramAlerter, '_validate_bot_token', return_value=True):
            with patch.object(TelegramAlerter, '_validate_chat_id', return_value=True):
                alerter = TelegramAlerter(
                    "TEST_BOT_123456789:FAKE_TEST_HASH_ABCDEFGHIJKLMNOPQRSTUV",
                    "TEST_CHAT_999999",
                    rate_limit=60
                )
        
        assert alerter.bot_token == "TEST_BOT_123456789:FAKE_TEST_HASH_ABCDEFGHIJKLMNOPQRSTUV"
        assert alerter.chat_id == "TEST_CHAT_999999"
        assert alerter.rate_limit == 60
        assert alerter.api_url == "https://api.telegram.org/botTEST_BOT_123456789:FAKE_TEST_HASH_ABCDEFGHIJKLMNOPQRSTUV"
        assert len(alerter.message_times) == 0
        assert len(alerter.message_queue) == 0
    
    def test_escape_html(self, alerter):
        """Test HTML escaping."""
        test_cases = [
            ("", ""),
            ("normal text", "normal text"),
            ("<script>alert('xss')</script>", "&lt;script&gt;alert('xss')&lt;/script&gt;"),
            ("key & value", "key &amp; value"),
            ('"quoted"', "&quot;quoted&quot;"),
            (None, "")
        ]
        
        for input_text, expected in test_cases:
            assert alerter._escape_html(input_text) == expected
    
    def test_format_alert(self, alerter):
        """Test alert message formatting."""
        message = alerter._format_alert(
            check_id="S1-health",
            status="fail",
            message="Service unavailable",
            severity=AlertSeverity.CRITICAL,
            details={"error": "timeout", "attempts": 3},
            latency_ms=250.5
        )
        
        # Check key components are present
        assert "🚨 CRITICAL: Veris Memory Alert" in message
        assert "Check:</b> S1-health" in message
        assert "Status:</b> FAIL ❌" in message
        assert "Latency:</b> 250.5ms" in message
        assert "Service unavailable" in message
        assert "error: timeout" in message
        assert "attempts: 3" in message
        assert "Action Required:" in message
    
    def test_format_alert_severity_emojis(self, alerter):
        """Test correct emoji usage for different severities."""
        test_cases = [
            (AlertSeverity.CRITICAL, "🚨"),
            (AlertSeverity.HIGH, "⚠️"),
            (AlertSeverity.WARNING, "⚡"),
            (AlertSeverity.INFO, "ℹ️")
        ]
        
        for severity, emoji in test_cases:
            message = alerter._format_alert(
                check_id="test",
                status="pass",
                message="test",
                severity=severity
            )
            assert emoji in message
    
    def test_format_summary(self, alerter):
        """Test summary message formatting."""
        message = alerter._format_summary(
            period_hours=24,
            total_checks=1440,
            passed_checks=1420,
            failed_checks=20,
            top_failures=[
                {"check_id": "S8-capacity", "count": 10},
                {"check_id": "S3-paraphrase", "count": 5}
            ],
            avg_latency_ms=45.2,
            uptime_percent=98.6
        )
        
        # Check key components
        assert "📊 Veris Sentinel Report" in message
        assert "Last 24 hours" in message
        assert "1,440" in message
        assert "1,420" in message
        assert "98.6%" in message
        assert "S8-capacity: 10 failures" in message
        assert "45.2ms" in message
    
    @pytest.mark.asyncio
    async def test_check_rate_limit(self, alerter):
        """Test rate limiting logic."""
        # Initially should allow messages
        assert await alerter._check_rate_limit() is True
        
        # Fill up to rate limit
        for _ in range(29):
            await alerter._check_rate_limit()
        
        # Should still allow one more
        assert await alerter._check_rate_limit() is True
        
        # Now should be rate limited
        assert await alerter._check_rate_limit() is False
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_cleanup(self, alerter):
        """Test rate limit cleanup of old timestamps."""
        # Add old timestamps
        old_time = datetime.utcnow() - timedelta(minutes=2)
        alerter.message_times = [old_time] * 10
        
        # Add recent timestamps
        recent_time = datetime.utcnow()
        alerter.message_times.extend([recent_time] * 5)
        
        # Check rate limit should clean old timestamps
        result = await alerter._check_rate_limit()
        
        assert result is True
        # Should only have 5 recent + 1 new = 6 timestamps
        assert len(alerter.message_times) == 6
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, alerter):
        """Test successful message sending."""
        message = TelegramMessage(
            text="Test message",
            parse_mode="HTML",
            disable_notification=False
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"ok": True})
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await alerter._send_message(message)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_send_message_api_error(self, alerter):
        """Test message sending with API error."""
        message = TelegramMessage(text="Test")
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "ok": False,
                "description": "Bot token invalid"
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await alerter._send_message(message)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_send_message_http_error(self, alerter):
        """Test message sending with HTTP error."""
        message = TelegramMessage(text="Test")
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 400
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await alerter._send_message(message)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_send_message_exception(self, alerter):
        """Test message sending with exception."""
        message = TelegramMessage(text="Test")
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.side_effect = Exception("Network error")
            
            result = await alerter._send_message(message)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_send_alert(self, alerter):
        """Test send_alert method."""
        with patch.object(alerter, '_send_message', return_value=True) as mock_send:
            result = await alerter.send_alert(
                check_id="S1-health",
                status="fail",
                message="Test failure",
                severity=AlertSeverity.HIGH,
                details={"key": "value"},
                latency_ms=100.0
            )
            
            assert result is True
            mock_send.assert_called_once()
            
            # Check the message was formatted
            call_args = mock_send.call_args[0][0]
            assert isinstance(call_args, TelegramMessage)
            assert "S1-health" in call_args.text
            assert "Test failure" in call_args.text
    
    @pytest.mark.asyncio
    async def test_send_summary(self, alerter):
        """Test send_summary method."""
        with patch.object(alerter, '_send_message', return_value=True) as mock_send:
            result = await alerter.send_summary(
                period_hours=24,
                total_checks=100,
                passed_checks=95,
                failed_checks=5,
                top_failures=[],
                avg_latency_ms=50.0,
                uptime_percent=95.0
            )
            
            assert result is True
            mock_send.assert_called_once()
            
            # Check the message was formatted
            call_args = mock_send.call_args[0][0]
            assert isinstance(call_args, TelegramMessage)
            assert "Veris Sentinel Report" in call_args.text
            assert call_args.disable_notification is True  # Summaries are non-urgent
    
    @pytest.mark.asyncio
    async def test_process_queue(self, alerter):
        """Test queue processing."""
        # Add messages to queue
        alerter.message_queue = [
            TelegramMessage(text="Message 1"),
            TelegramMessage(text="Message 2"),
            TelegramMessage(text="Message 3")
        ]
        
        with patch.object(alerter, '_send_message', return_value=True) as mock_send:
            with patch.object(alerter, '_check_rate_limit', return_value=True):
                sent_count = await alerter.process_queue()
                
                assert sent_count == 3
                assert len(alerter.message_queue) == 0
                assert mock_send.call_count == 3
    
    @pytest.mark.asyncio
    async def test_process_queue_rate_limited(self, alerter):
        """Test queue processing with rate limiting."""
        alerter.message_queue = [
            TelegramMessage(text="Message 1"),
            TelegramMessage(text="Message 2")
        ]
        
        # First call allows, second doesn't
        rate_limit_returns = [True, False]
        
        with patch.object(alerter, '_send_message', return_value=True) as mock_send:
            with patch.object(alerter, '_check_rate_limit', side_effect=rate_limit_returns):
                sent_count = await alerter.process_queue()
                
                assert sent_count == 1
                assert len(alerter.message_queue) == 1  # One message left
                assert mock_send.call_count == 1
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, alerter):
        """Test connection testing success."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "ok": True,
                "result": {"username": "test_bot"}
            })
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await alerter.test_connection()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, alerter):
        """Test connection testing failure."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 401
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await alerter.test_connection()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_rate_limit_queue_interaction(self, alerter):
        """Test interaction between rate limiting and queueing."""
        # Set up rate limiting to be exceeded
        alerter.rate_limit = 2
        
        with patch.object(alerter, '_send_message', return_value=True):
            # Send messages up to rate limit
            await alerter.send_alert("test1", "fail", "msg1", AlertSeverity.HIGH)
            await alerter.send_alert("test2", "fail", "msg2", AlertSeverity.HIGH)
            
            # This should be queued
            result = await alerter.send_alert("test3", "fail", "msg3", AlertSeverity.HIGH)
            
            assert result is False  # Indicates it was queued
            assert len(alerter.message_queue) == 1
    
    def test_telegram_message_dataclass(self):
        """Test TelegramMessage dataclass."""
        msg = TelegramMessage(
            text="Test",
            parse_mode="Markdown",
            disable_web_page_preview=False,
            disable_notification=True
        )
        
        assert msg.text == "Test"
        assert msg.parse_mode == "Markdown"
        assert msg.disable_web_page_preview is False
        assert msg.disable_notification is True
    
    def test_alert_severity_enum(self):
        """Test AlertSeverity enum."""
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.INFO.value == "info"
    
    def test_token_validation(self):
        """Test bot token validation."""
        with patch.object(TelegramAlerter, '_validate_bot_token', return_value=True):
            with patch.object(TelegramAlerter, '_validate_chat_id', return_value=True):
                alerter = TelegramAlerter(
                    "TEST_BOT_123456789:FAKE_TEST_HASH_ABCDEFGHIJKLMNOPQRSTUV",
                    "TEST_CHAT_999999",
                    rate_limit=30
                )
        
        # Test validation directly
        # Valid tokens
        assert alerter._validate_bot_token("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk") is True
        
        # Invalid tokens
        assert alerter._validate_bot_token("YOUR_BOT_TOKEN_HERE") is False
        assert alerter._validate_bot_token("test_token") is False
        assert alerter._validate_bot_token("invalid") is False
        assert alerter._validate_bot_token("") is False
        assert alerter._validate_bot_token(None) is False
    
    def test_chat_id_validation(self):
        """Test chat ID validation."""
        with patch.object(TelegramAlerter, '_validate_bot_token', return_value=True):
            with patch.object(TelegramAlerter, '_validate_chat_id', return_value=True):
                alerter = TelegramAlerter(
                    "TEST_BOT_123456789:FAKE_TEST_HASH_ABCDEFGHIJKLMNOPQRSTUV",
                    "TEST_CHAT_999999",
                    rate_limit=30
                )
        
        # Test validation directly
        # Valid chat IDs
        assert alerter._validate_chat_id("123456789") is True
        assert alerter._validate_chat_id("-123456789") is True
        
        # Invalid chat IDs
        assert alerter._validate_chat_id("YOUR_CHAT_ID_HERE") is False
        assert alerter._validate_chat_id("test_chat") is False
        assert alerter._validate_chat_id("abc") is False
        assert alerter._validate_chat_id("") is False
        assert alerter._validate_chat_id(None) is False