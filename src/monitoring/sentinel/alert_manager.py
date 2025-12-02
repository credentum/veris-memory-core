#!/usr/bin/env python3
"""
Alert Manager for Veris Sentinel

This module manages alert deduplication, routing, and delivery to various
channels including Telegram and GitHub.

Author: Workspace 002
Date: 2025-08-19
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Deque
from collections import defaultdict, deque
import aiohttp

from .telegram_alerter import TelegramAlerter, AlertSeverity
from .webhook_alerter import GitHubWebhookAlerter
from .models import CheckResult

logger = logging.getLogger(__name__)


class AlertDeduplicator:
    """
    Deduplicates alerts to prevent notification spam.
    
    Uses a sliding window approach to track recent alerts and prevent
    duplicate notifications within a configurable time window.
    """
    
    def __init__(self, window_minutes: int = 30):
        """
        Initialize deduplicator.
        
        Args:
            window_minutes: Deduplication window in minutes
        """
        self.window_minutes = window_minutes
        self.alert_history: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def should_alert(self, alert_key: str) -> bool:
        """
        Check if an alert should be sent.
        
        Args:
            alert_key: Unique identifier for the alert type
        
        Returns:
            True if alert should be sent, False if it's a duplicate
        """
        async with self.lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=self.window_minutes)
            
            # Clean old entries
            self.alert_history[alert_key] = [
                t for t in self.alert_history[alert_key] if t > cutoff
            ]
            
            # Check if we've sent this alert recently
            if self.alert_history[alert_key]:
                logger.debug(f"Alert {alert_key} deduplicated (sent {len(self.alert_history[alert_key])} times in window)")
                return False
            
            # Record this alert
            self.alert_history[alert_key].append(now)
            return True
    
    def get_alert_key(self, check_id: str, status: str, message: str) -> str:
        """
        Generate a unique key for an alert.
        
        Args:
            check_id: Check identifier
            status: Check status
            message: Alert message
        
        Returns:
            Hash key for the alert
        """
        # Create a hash of the essential alert components
        content = f"{check_id}:{status}:{message[:100]}"  # Use first 100 chars of message
        return hashlib.md5(content.encode()).hexdigest()


class GitHubIssueCreator:
    """
    Creates GitHub issues for critical alerts with rate limiting.
    """
    
    def __init__(self, token: str, repo: str, labels: List[str] = None, rate_limit: int = 10):
        """
        Initialize GitHub issue creator.
        
        Args:
            token: GitHub API token
            repo: Repository in format "owner/repo"
            labels: Default labels for issues
            rate_limit: Max issues per hour (default: 10)
        """
        self.token = token
        self.repo = repo
        self.labels = labels or ["sentinel", "automated", "monitoring"]
        self.api_url = f"https://api.github.com/repos/{repo}/issues"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.rate_limit = rate_limit
        self.issue_times: Deque[datetime] = deque(maxlen=rate_limit)
        self.lock = asyncio.Lock()
    
    async def create_issue(
        self,
        title: str,
        body: str,
        labels: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a GitHub issue with rate limiting.
        
        Args:
            title: Issue title
            body: Issue body (markdown)
            labels: Additional labels
        
        Returns:
            Issue URL if created successfully
        """
        # Check rate limit
        async with self.lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(hours=1)
            
            # Clean old entries
            while self.issue_times and self.issue_times[0] < cutoff:
                self.issue_times.popleft()
            
            # Check if we've hit rate limit
            if len(self.issue_times) >= self.rate_limit:
                logger.warning(f"GitHub API rate limit reached ({self.rate_limit} issues/hour)")
                return None
            
            # Record this issue creation
            self.issue_times.append(now)
        
        try:
            all_labels = self.labels.copy()
            if labels:
                all_labels.extend(labels)
            
            payload = {
                "title": title,
                "body": body,
                "labels": all_labels
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        issue_url = result.get("html_url")
                        logger.info(f"Created GitHub issue: {issue_url}")
                        return issue_url
                    else:
                        logger.error(f"Failed to create GitHub issue: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error creating GitHub issue: {e}")
            return None


class AlertManager:
    """
    Central alert management system for Sentinel.
    
    Coordinates alert deduplication, severity-based routing, and delivery
    to multiple channels.
    """
    
    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        github_token: Optional[str] = None,
        github_repo: Optional[str] = None,
        dedup_window_minutes: int = 30,
        alert_threshold_failures: int = 3
    ):
        """
        Initialize alert manager.
        
        Args:
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
            github_token: GitHub API token
            github_repo: GitHub repository
            dedup_window_minutes: Deduplication window
            alert_threshold_failures: Failures before alerting
        """
        # Initialize Telegram if configured
        self.telegram = None
        if telegram_token and telegram_chat_id:
            try:
                self.telegram = TelegramAlerter(telegram_token, telegram_chat_id)
                # Don't log sensitive tokens
                logger.info("Telegram alerting enabled (credentials loaded)")
            except ValueError as e:
                logger.warning(f"Telegram alerting disabled: {e}")
                self.telegram = None
        
        # Initialize GitHub Issue Creator if configured
        self.github = None
        if github_token and github_repo:
            self.github = GitHubIssueCreator(github_token, github_repo)
            # Don't log sensitive tokens
            logger.info(f"GitHub issue creation enabled for repo: {github_repo}")
        
        # Initialize GitHub Webhook Alerter (uses environment variables)
        self.webhook_alerter = GitHubWebhookAlerter()
        webhook_status = self.webhook_alerter.get_status()
        if webhook_status['enabled']:
            logger.info(f"GitHub webhook alerting enabled for repo: {webhook_status['github_repo']}")
        else:
            logger.warning("GitHub webhook alerting disabled (no GITHUB_TOKEN found)")
        
        # Initialize deduplicator
        self.deduplicator = AlertDeduplicator(dedup_window_minutes)
        self.alert_threshold_failures = alert_threshold_failures
        
        # Track failure counts for threshold alerting
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.failure_lock = asyncio.Lock()
        
        logger.info("Alert manager initialized with configured services")
    
    async def process_check_result(self, result: CheckResult) -> None:
        """
        Process a check result and send alerts if needed.
        
        Args:
            result: Check result to process
        """
        # Determine severity
        severity = self._determine_severity(result)
        
        # Update failure tracking
        await self._update_failure_tracking(result)
        
        # Check if we should alert
        if not await self._should_alert(result, severity):
            return
        
        # Send alerts based on severity
        await self._route_alert(result, severity)
    
    def _determine_severity(self, result: CheckResult) -> AlertSeverity:
        """
        Determine alert severity based on check result.
        
        Args:
            result: Check result
        
        Returns:
            Alert severity level
        """
        if result.status == "pass":
            return AlertSeverity.INFO
        
        # Map check IDs to severity levels
        critical_checks = ["S1-health-probes", "S5-security-negatives", "S6-backup-restore"]
        high_checks = ["S2-golden-fact-recall", "S8-capacity-smoke", "S4-metrics-wiring"]
        
        if result.check_id in critical_checks:
            return AlertSeverity.CRITICAL if result.status == "fail" else AlertSeverity.HIGH
        elif result.check_id in high_checks:
            return AlertSeverity.HIGH if result.status == "fail" else AlertSeverity.WARNING
        else:
            return AlertSeverity.WARNING if result.status == "fail" else AlertSeverity.INFO
    
    async def _update_failure_tracking(self, result: CheckResult) -> None:
        """Update failure counts for threshold tracking."""
        async with self.failure_lock:
            if result.status == "fail":
                self.failure_counts[result.check_id] += 1
            else:
                # Reset on success
                self.failure_counts[result.check_id] = 0
    
    async def _should_alert(self, result: CheckResult, severity: AlertSeverity) -> bool:
        """
        Determine if an alert should be sent.
        
        Args:
            result: Check result
            severity: Alert severity
        
        Returns:
            True if alert should be sent
        """
        # Always skip info level for individual alerts
        if severity == AlertSeverity.INFO:
            return False
        
        # Check failure threshold
        if result.status == "fail":
            failure_count = self.failure_counts.get(result.check_id, 0)
            if failure_count < self.alert_threshold_failures:
                logger.debug(f"Check {result.check_id} has {failure_count} failures, threshold is {self.alert_threshold_failures}")
                return False
        
        # Check deduplication
        alert_key = self.deduplicator.get_alert_key(
            result.check_id,
            result.status,
            result.message
        )
        
        return await self.deduplicator.should_alert(alert_key)
    
    async def _route_alert(self, result: CheckResult, severity: AlertSeverity) -> None:
        """
        Route alert to appropriate channels based on severity.
        
        Args:
            result: Check result
            severity: Alert severity
        """
        tasks = []
        
        # Send to Telegram
        if self.telegram:
            tasks.append(self._send_telegram_alert(result, severity))
        
        # Send to GitHub Actions via webhook (for all alert levels)
        if self.webhook_alerter and self.webhook_alerter.get_status()['enabled']:
            tasks.append(self._send_webhook_alert(result, severity))
        
        # Create GitHub issue for critical alerts (legacy method)
        if self.github and severity == AlertSeverity.CRITICAL:
            tasks.append(self._create_github_issue(result))
        
        # Execute all tasks concurrently with retry for critical alerts
        if tasks:
            if severity == AlertSeverity.CRITICAL:
                # Retry critical alerts with exponential backoff
                await self._execute_with_retry(tasks, max_retries=3)
            else:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_telegram_alert(self, result: CheckResult, severity: AlertSeverity) -> None:
        """Send alert to Telegram."""
        try:
            success = await self.telegram.send_alert(
                check_id=result.check_id,
                status=result.status,
                message=result.message,
                severity=severity,
                details=result.details,
                latency_ms=result.latency_ms
            )
            
            if success:
                logger.info(f"Telegram alert sent for {result.check_id}")
            else:
                logger.error(f"Failed to send Telegram alert for {result.check_id}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    async def _send_webhook_alert(self, result: CheckResult, severity: AlertSeverity) -> None:
        """Send alert to GitHub Actions via webhook."""
        try:
            success = await self.webhook_alerter.send_alert(result)
            
            if success:
                logger.info(f"GitHub webhook alert sent for {result.check_id}")
            else:
                logger.error(f"Failed to send GitHub webhook alert for {result.check_id}")
                
        except Exception as e:
            logger.error(f"Error sending GitHub webhook alert: {e}")
    
    async def _create_github_issue(self, result: CheckResult) -> None:
        """Create GitHub issue for critical alert."""
        try:
            title = f"[Sentinel] Critical: {result.check_id} failure"
            
            body = f"""## Sentinel Critical Alert

**Check:** {result.check_id}
**Status:** {result.status}
**Time:** {result.timestamp.isoformat()}
**Latency:** {result.latency_ms:.1f}ms

### Message
{result.message}

### Details
```json
{json.dumps(result.details, indent=2) if result.details else "No additional details"}
```

### Action Required
This is a critical alert requiring immediate investigation.

---
*This issue was automatically created by Veris Sentinel monitoring system.*
"""
            
            labels = ["critical", result.check_id.split("-")[0].lower()]  # e.g., ["critical", "s1"]
            
            issue_url = await self.github.create_issue(title, body, labels)
            
            if issue_url:
                logger.info(f"GitHub issue created: {issue_url}")
                
                # Send follow-up Telegram with issue link
                if self.telegram:
                    await self.telegram.send_alert(
                        check_id=result.check_id,
                        status="github_issue",
                        message=f"GitHub issue created: {issue_url}",
                        severity=AlertSeverity.INFO
                    )
            else:
                logger.error(f"Failed to create GitHub issue for {result.check_id}")
                
        except Exception as e:
            logger.error(f"Error creating GitHub issue: {e}")
    
    async def send_summary(
        self,
        period_hours: int,
        check_results: List[CheckResult]
    ) -> None:
        """
        Send a periodic summary.
        
        Args:
            period_hours: Summary period in hours
            check_results: List of check results in the period
        """
        if not self.telegram or not check_results:
            return
        
        # Calculate statistics
        total_checks = len(check_results)
        passed_checks = sum(1 for r in check_results if r.status == "pass")
        failed_checks = sum(1 for r in check_results if r.status == "fail")
        
        # Calculate top failures
        failure_counts = defaultdict(int)
        for result in check_results:
            if result.status == "fail":
                failure_counts[result.check_id] += 1
        
        top_failures = [
            {"check_id": check_id, "count": count}
            for check_id, count in sorted(
                failure_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        # Calculate average latency
        latencies = [r.latency_ms for r in check_results if r.latency_ms is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # Calculate uptime
        uptime_percent = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        # Send summary
        try:
            await self.telegram.send_summary(
                period_hours=period_hours,
                total_checks=total_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                top_failures=top_failures,
                avg_latency_ms=avg_latency,
                uptime_percent=uptime_percent
            )
            logger.info(f"Sent {period_hours}-hour summary to Telegram")
        except Exception as e:
            logger.error(f"Error sending summary: {e}")
    
    async def test_alerting(self) -> Dict[str, bool]:
        """
        Test all configured alerting channels.
        
        Returns:
            Dictionary of channel test results
        """
        results = {}
        
        # Test Telegram
        if self.telegram:
            try:
                results["telegram"] = await self.telegram.test_connection()
            except Exception as e:
                logger.error(f"Telegram test failed: {e}")
                results["telegram"] = False
        
        # Test GitHub
        if self.github:
            try:
                # Test by checking if repo is accessible
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.github.com/repos/{self.github.repo}"
                    async with session.get(url, headers=self.github.headers) as response:
                        results["github"] = response.status == 200
            except Exception as e:
                logger.error(f"GitHub test failed: {e}")
                results["github"] = False
        
        return results
    
    async def _execute_with_retry(
        self,
        tasks: List,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> None:
        """
        Execute tasks with exponential backoff retry for critical alerts.
        
        Args:
            tasks: List of async tasks to execute
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
        """
        for attempt in range(max_retries):
            try:
                results = await asyncio.gather(*tasks, return_exceptions=False)
                # If successful, return
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Critical alert delivery failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    # Recreate tasks for retry
                    continue
                else:
                    logger.error(f"Critical alert delivery failed after {max_retries} attempts: {e}")
                    raise
