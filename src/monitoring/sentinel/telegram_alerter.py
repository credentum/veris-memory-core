#!/usr/bin/env python3
"""
Telegram Alerter for Veris Sentinel

This module provides Telegram bot integration for sending alerts from the
Veris Memory Sentinel monitoring system.

Author: Workspace 002
Date: 2025-08-19
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import aiohttp
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    WARNING = "warning"
    INFO = "info"


@dataclass
class TelegramMessage:
    """Structured Telegram message."""
    text: str
    parse_mode: str = "HTML"
    disable_web_page_preview: bool = True
    disable_notification: bool = False


class TelegramAlerter:
    """
    Telegram bot alerter for Sentinel monitoring.
    
    Handles sending alerts to Telegram with rate limiting,
    formatting, and error handling.
    """
    
    # Emoji mappings for severity levels
    SEVERITY_EMOJIS = {
        AlertSeverity.CRITICAL: "🚨",
        AlertSeverity.HIGH: "⚠️",
        AlertSeverity.WARNING: "⚡",
        AlertSeverity.INFO: "ℹ️"
    }
    
    # Status emojis
    STATUS_EMOJIS = {
        "pass": "✅",
        "fail": "❌",
        "error": "🔥",
        "timeout": "⏱️",
        "unknown": "❓"
    }
    
    def __init__(self, bot_token: str, chat_id: str, rate_limit: int = 30) -> None:
        """
        Initialize Telegram alerter.

        Args:
            bot_token: Telegram bot API token
            chat_id: Target chat/channel ID
            rate_limit: Max messages per minute (default: 30)
        """
        # Validate bot token format
        if not self._validate_bot_token(bot_token):
            raise ValueError("Invalid bot token format")
        
        # Validate chat ID format
        if not self._validate_chat_id(chat_id):
            raise ValueError("Invalid chat ID format")
        
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.rate_limit = rate_limit
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Rate limiting
        self.message_times: List[datetime] = []
        self.rate_limit_lock = asyncio.Lock()
        
        # Message queue for batching
        self.message_queue: List[TelegramMessage] = []
        self.queue_lock = asyncio.Lock()
        
        # Log initialization without exposing sensitive data
        logger.info(f"Telegram alerter initialized for chat [REDACTED]")
    
    async def send_alert(
        self,
        check_id: str,
        status: str,
        message: str,
        severity: AlertSeverity,
        details: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None
    ) -> bool:
        """
        Send an alert to Telegram.
        
        Args:
            check_id: Check identifier (e.g., "S1-health-probes")
            status: Check status (pass/fail/error/timeout)
            message: Alert message
            severity: Alert severity level
            details: Additional details dictionary
            latency_ms: Check latency in milliseconds
        
        Returns:
            True if message sent successfully
        """
        formatted_message = self._format_alert(
            check_id, status, message, severity, details, latency_ms
        )
        
        # Disable notifications for info level
        disable_notification = severity == AlertSeverity.INFO
        
        telegram_msg = TelegramMessage(
            text=formatted_message,
            disable_notification=disable_notification
        )
        
        return await self._send_message(telegram_msg)
    
    async def send_summary(
        self,
        period_hours: int,
        total_checks: int,
        passed_checks: int,
        failed_checks: int,
        top_failures: List[Dict[str, Any]],
        avg_latency_ms: float,
        uptime_percent: float
    ) -> bool:
        """
        Send a periodic summary to Telegram.
        
        Args:
            period_hours: Summary period in hours
            total_checks: Total checks executed
            passed_checks: Number of passed checks
            failed_checks: Number of failed checks
            top_failures: List of top failure details
            avg_latency_ms: Average check latency
            uptime_percent: System uptime percentage
        
        Returns:
            True if message sent successfully
        """
        formatted_message = self._format_summary(
            period_hours, total_checks, passed_checks, failed_checks,
            top_failures, avg_latency_ms, uptime_percent
        )
        
        telegram_msg = TelegramMessage(
            text=formatted_message,
            disable_notification=True  # Summaries are non-urgent
        )
        
        return await self._send_message(telegram_msg)
    
    def _format_alert(
        self,
        check_id: str,
        status: str,
        message: str,
        severity: AlertSeverity,
        details: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None
    ) -> str:
        """Format an alert message for Telegram."""
        severity_emoji = self.SEVERITY_EMOJIS.get(severity, "")
        status_emoji = self.STATUS_EMOJIS.get(status, "❓")

        # Build header
        header = f"<b>{severity_emoji} {severity.value.upper()}: Veris Memory Alert</b>"

        # Build body - escape all user-controlled data to prevent HTML injection
        lines = [
            header,
            "━━━━━━━━━━━━━━━━━━━━━",
            f"<b>Check:</b> {self._escape_html(check_id)}",
            f"<b>Status:</b> {self._escape_html(status.upper())} {status_emoji}",
            f"<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        ]

        if latency_ms:
            lines.append(f"<b>Latency:</b> {latency_ms:.1f}ms")

        lines.append("")
        lines.append(f"<b>Message:</b>\n{self._escape_html(message)}")

        # Add filtered details if provided (reduce verbosity for Telegram)
        if details:
            filtered = self._filter_alert_details(details)
            if filtered:
                lines.append("")
                lines.append("<b>Details:</b>")
                for key, value in filtered.items():
                    if isinstance(value, (list, dict)):
                        # Recursively escape HTML BEFORE JSON dumping
                        value = self._escape_nested_html(value)
                        value = json.dumps(value, indent=2)
                    lines.append(f"• {self._escape_html(str(key))}: {self._escape_html(str(value))}")

        # Add action required for critical/high severity
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            lines.append("")
            lines.append("<b>Action Required:</b> Immediate investigation")

        lines.append("━━━━━━━━━━━━━━━━━━━━━")

        return "\n".join(lines)

    def _filter_alert_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter alert details to reduce verbosity for Telegram.

        Prioritizes failure-specific information and removes verbose data
        like full test_results arrays that can exceed Telegram's 4096 char limit.

        Args:
            details: Full details dictionary from CheckResult

        Returns:
            Filtered dictionary with only relevant alert information
        """
        # Keys to EXCLUDE (verbose, not useful for quick alerts)
        exclude_keys = {
            "test_results",           # Full array of all test results (very verbose)
            "passed_test_names",      # Not needed for failure alerts
            "configuration_baseline", # Config reference, not failure info
            "endpoint_checks",        # Verbose endpoint status
            "env_status",             # Full env var listing
            "file_status",            # Full file status listing
            "security_checks",        # Verbose security details
            "resource_info",          # Verbose resource details
            "version_info",           # Verbose version details per package
            "db_checks",              # Verbose DB check details
        }

        # Keys to PRIORITIZE (failure-specific, useful for debugging)
        priority_keys = [
            "failed_tests",           # Count of failures
            "failed_test_names",      # Which tests failed (keep for failures)
            "config_issues",          # Specific config problems
            "version_issues",         # Specific version problems
            "security_issues",        # Specific security problems
            "resource_issues",        # Specific resource problems
            "violations",             # Policy violations
            "missing_critical",       # Missing critical items
            "issues",                 # Generic issues list
            "error",                  # Error message
            "error_type",             # Error type
        ]

        filtered = {}

        # First, add priority keys if they have content
        for key in priority_keys:
            if key in details:
                value = details[key]
                # Only include if non-empty
                if value and (not isinstance(value, (list, dict)) or len(value) > 0):
                    # For failed_test_names, include it (useful context)
                    if key == "failed_test_names":
                        filtered[key] = value
                    else:
                        filtered[key] = value

        # Then add other non-excluded keys with limits
        for key, value in details.items():
            if key in exclude_keys or key in filtered:
                continue

            # Skip empty values
            if not value and value != 0:
                continue

            # Limit list lengths
            if isinstance(value, list) and len(value) > 5:
                filtered[key] = value[:5]
                filtered[f"{key}_truncated"] = f"(showing 5 of {len(value)})"
            # Limit dict complexity
            elif isinstance(value, dict) and len(value) > 5:
                # Just show key count for complex dicts
                filtered[key] = f"({len(value)} items)"
            else:
                filtered[key] = value

        return filtered
    
    def _format_summary(
        self,
        period_hours: int,
        total_checks: int,
        passed_checks: int,
        failed_checks: int,
        top_failures: List[Dict[str, Any]],
        avg_latency_ms: float,
        uptime_percent: float
    ) -> str:
        """Format a summary message for Telegram."""
        period_text = f"{period_hours} hours" if period_hours != 24 else "24 hours"
        pass_percent = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        lines = [
            "<b>📊 Veris Sentinel Report</b>",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"<b>Period:</b> Last {period_text}",
            f"<b>Total Checks:</b> {total_checks:,}",
            f"✅ <b>Passed:</b> {passed_checks:,} ({pass_percent:.1f}%)",
            f"❌ <b>Failed:</b> {failed_checks:,} ({100-pass_percent:.1f}%)"
        ]
        
        if top_failures:
            lines.append("")
            lines.append("<b>Top Issues:</b>")
            for i, failure in enumerate(top_failures[:5], 1):
                check_id = failure.get('check_id', 'Unknown')
                count = failure.get('count', 0)
                # Escape check_id to prevent HTML injection
                lines.append(f"{i}. {self._escape_html(check_id)}: {count} failures")
        
        lines.extend([
            "",
            f"<b>Avg Response Time:</b> {avg_latency_ms:.1f}ms",
            f"<b>Uptime:</b> {uptime_percent:.1f}%",
            "━━━━━━━━━━━━━━━━━━━━━"
        ])
        
        return "\n".join(lines)
    
    async def _send_message(self, message: TelegramMessage) -> bool:
        """
        Send a message to Telegram with rate limiting.
        
        Args:
            message: TelegramMessage object
        
        Returns:
            True if sent successfully
        """
        # Check rate limit
        if not await self._check_rate_limit():
            logger.warning("Rate limit exceeded, queuing message")
            async with self.queue_lock:
                self.message_queue.append(message)
            return False
        
        # Validate and truncate message text if needed (Telegram max: 4096 characters)
        message_text = message.text
        if len(message_text) > 4096:
            logger.warning(f"Telegram message truncated from {len(message_text)} to 4096 characters")
            message_text = message_text[:4093] + "..."

        # Send via Telegram API
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_url}/sendMessage"
                # Telegram API requires form-data or application/x-www-form-urlencoded
                # NOT application/json, so we use data= instead of json=
                params = {
                    "chat_id": self.chat_id,
                    "text": message_text,
                    "parse_mode": message.parse_mode,
                    "disable_web_page_preview": str(message.disable_web_page_preview).lower(),
                    "disable_notification": str(message.disable_notification).lower()
                }

                # Explicitly set Content-Type header for clarity
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded"
                }

                async with session.post(url, data=params, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            logger.info("Telegram message sent successfully")
                            return True
                        else:
                            logger.error(f"Telegram API error: {result.get('description')}")
                            return False
                    else:
                        logger.error(f"Telegram HTTP error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limit."""
        async with self.rate_limit_lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=1)
            
            # Remove old timestamps
            self.message_times = [t for t in self.message_times if t > cutoff]
            
            # Check if we can send
            if len(self.message_times) < self.rate_limit:
                self.message_times.append(now)
                return True
            
            return False
    
    async def process_queue(self) -> int:
        """
        Process queued messages.
        
        Returns:
            Number of messages sent from queue
        """
        sent_count = 0
        
        async with self.queue_lock:
            while self.message_queue and await self._check_rate_limit():
                message = self.message_queue.pop(0)
                if await self._send_message(message):
                    sent_count += 1
                await asyncio.sleep(0.1)  # Small delay between messages
        
        if sent_count > 0:
            logger.info(f"Processed {sent_count} queued messages")
        
        return sent_count
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""

        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def _escape_nested_html(self, obj: Any, depth: int = 0, max_depth: int = 10, visited: Optional[set] = None) -> Any:
        """
        Recursively escape HTML in nested data structures.

        This prevents Telegram HTML parsing errors when alert details
        contain angle brackets (< >) that would be interpreted as HTML tags.

        Args:
            obj: Any object (dict, list, str, or other)
            depth: Current recursion depth (for circular reference protection)
            max_depth: Maximum recursion depth allowed (default: 10)
            visited: Set of object IDs already visited (circular reference detection)

        Returns:
            The same structure with all strings HTML-escaped

        Raises:
            ValueError: If max recursion depth is exceeded or circular reference detected
        """
        # Prevent infinite recursion
        if depth > max_depth:
            logger.warning(f"Max recursion depth {max_depth} exceeded in _escape_nested_html")
            return "[MAX_DEPTH_EXCEEDED]"

        # Initialize visited set for circular reference detection
        if visited is None:
            visited = set()

        # Check for circular references (only for mutable objects)
        if isinstance(obj, (dict, list)):
            obj_id = id(obj)
            if obj_id in visited:
                logger.warning("Circular reference detected in _escape_nested_html")
                return "[CIRCULAR_REF]"
            visited.add(obj_id)

        try:
            if isinstance(obj, dict):
                # Handle dict comprehension with error handling for non-string keys
                result = {}
                for k, v in obj.items():
                    try:
                        safe_key = self._escape_html(str(k))
                        safe_value = self._escape_nested_html(v, depth + 1, max_depth, visited)
                        result[safe_key] = safe_value
                    except Exception as e:
                        logger.error(f"Error escaping dict key/value: {e}")
                        result[str(k)] = "[ERROR]"
                return result

            elif isinstance(obj, list):
                return [self._escape_nested_html(item, depth + 1, max_depth, visited) for item in obj]

            elif isinstance(obj, str):
                return self._escape_html(obj)

            elif obj is None:
                return None

            else:
                # For numbers, booleans, etc., convert to string and escape
                try:
                    return self._escape_html(str(obj))
                except Exception as e:
                    logger.error(f"Error converting object to string: {e}")
                    return "[ERROR]"

        except Exception as e:
            logger.error(f"Unexpected error in _escape_nested_html: {e}")
            return "[ERROR]"

        finally:
            # Remove from visited set after processing (for proper cleanup)
            if isinstance(obj, (dict, list)):
                visited.discard(id(obj))

    async def test_connection(self) -> bool:
        """
        Test Telegram bot connection.
        
        Returns:
            True if bot is accessible and configured correctly
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_url}/getMe"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            bot_info = result.get("result", {})
                            logger.info(f"Connected to Telegram bot: @{bot_info.get('username')}")
                            return True
                        else:
                            logger.error(f"Telegram bot error: {result.get('description')}")
                            return False
                    else:
                        logger.error(f"Telegram connection failed: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to test Telegram connection: {e}")
            return False
    
    def _validate_bot_token(self, token: str) -> bool:
        """
        Validate Telegram bot token format.
        
        Args:
            token: Bot token to validate
        
        Returns:
            True if valid, False otherwise
        """
        if not token or not isinstance(token, str):
            return False
        
        # Check for placeholder values
        placeholders = [
            "YOUR_BOT_TOKEN_HERE",
            "YOUR_TOKEN_HERE",
            "BOT_TOKEN",
            "test_token",
            "example_token"
        ]
        if token.upper() in [p.upper() for p in placeholders]:
            return False
        
        # Telegram bot tokens have format: <bot_id>:<hash>
        if ":" not in token:
            return False
        
        parts = token.split(":")
        if len(parts) != 2:
            return False
        
        # Bot ID should be numeric (usually 10 digits)
        if not parts[0].isdigit() or len(parts[0]) < 8:
            return False
        
        # Hash should be alphanumeric with possible - and _
        # and should be at least 35 characters
        if len(parts[1]) < 35:
            return False
        
        if not all(c.isalnum() or c in "-_" for c in parts[1]):
            return False
        
        return True
    
    def _validate_chat_id(self, chat_id: str) -> bool:
        """
        Validate Telegram chat ID format.
        
        Args:
            chat_id: Chat ID to validate
        
        Returns:
            True if valid, False otherwise
        """
        if not chat_id or not isinstance(chat_id, str):
            return False
        
        # Check for placeholder values
        if chat_id.upper() in ["YOUR_CHAT_ID_HERE", "CHAT_ID", "test_chat"]:
            return False
        
        # Chat ID should be numeric or start with - for groups
        if chat_id.startswith("-"):
            # Group chat ID
            return chat_id[1:].isdigit() and len(chat_id) > 5
        else:
            # Private chat ID
            return chat_id.isdigit() and len(chat_id) > 5
