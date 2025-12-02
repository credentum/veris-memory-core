#!/usr/bin/env python3
"""
GitHub Webhook Alerter for Veris Sentinel

This module provides GitHub Actions integration for sending alerts from the
Veris Memory Sentinel monitoring system via repository dispatch events.

Author: Claude Code Integration
Date: 2025-08-20
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import aiohttp
from dataclasses import asdict

from .models import CheckResult

logger = logging.getLogger(__name__)


class GitHubWebhookAlerter:
    """
    GitHub webhook alerter for Sentinel monitoring.
    
    Sends alerts to GitHub Actions via repository dispatch events,
    enabling automated response to monitoring failures.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize GitHub webhook alerter."""
        self.config = config or {}
        
        # GitHub configuration
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv('SENTINEL_GITHUB_REPO', 'credentum/veris-memory')
        self.webhook_secret = os.getenv('SENTINEL_WEBHOOK_SECRET')
        
        # Rate limiting
        self.max_alerts_per_minute = 10
        self.recent_alerts = []
        
        # GitHub API configuration
        self.github_api_base = "https://api.github.com"
        self.dispatch_url = f"{self.github_api_base}/repos/{self.github_repo}/dispatches"
        
        logger.info(f"GitHub webhook alerter initialized for repo: {self.github_repo}")
    
    async def send_alert(self, check_result: CheckResult) -> bool:
        """
        Send alert to GitHub Actions via repository dispatch.
        
        Args:
            check_result: The check result to send as an alert
            
        Returns:
            bool: True if alert sent successfully, False otherwise
        """
        if not self._should_send_alert(check_result):
            return False
        
        try:
            # Prepare payload for GitHub Actions
            payload = self._prepare_github_payload(check_result)
            
            # Send to GitHub
            success = await self._send_repository_dispatch(payload)
            
            if success:
                self._record_alert_sent(check_result)
                logger.info(f"GitHub alert sent successfully for {check_result.check_id}")
            else:
                logger.error(f"Failed to send GitHub alert for {check_result.check_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending GitHub alert: {str(e)}")
            return False
    
    def _should_send_alert(self, check_result: CheckResult) -> bool:
        """
        Determine if alert should be sent based on configuration and rate limiting.
        
        Args:
            check_result: The check result to evaluate
            
        Returns:
            bool: True if alert should be sent
        """
        # Check if GitHub integration is configured
        if not self.github_token:
            logger.warning("GitHub token not configured, skipping GitHub alert")
            return False
        
        # Check rate limiting
        now = datetime.now()
        recent_cutoff = now.timestamp() - 60  # Last minute
        self.recent_alerts = [
            alert_time for alert_time in self.recent_alerts 
            if alert_time > recent_cutoff
        ]
        
        if len(self.recent_alerts) >= self.max_alerts_per_minute:
            logger.warning("Rate limit exceeded, skipping GitHub alert")
            return False
        
        # Only send for failures and warnings (not passing checks)
        if check_result.status == "pass":
            return False
        
        return True
    
    def _prepare_github_payload(self, check_result: CheckResult) -> Dict[str, Any]:
        """
        Prepare payload for GitHub repository dispatch.
        
        Args:
            check_result: The check result to convert
            
        Returns:
            Dict containing the payload for GitHub Actions
        """
        # Map severity levels
        severity_map = {
            "fail": "critical",
            "warn": "warning",
            "pass": "info"
        }
        
        severity = severity_map.get(check_result.status, "warning")
        
        # Prepare client payload for GitHub Actions
        client_payload = {
            "alert_id": f"{check_result.check_id}-{int(check_result.timestamp.timestamp())}",
            "check_id": check_result.check_id,
            "status": check_result.status,
            "severity": severity,
            "message": check_result.message,
            "timestamp": check_result.timestamp.isoformat(),
            "latency_ms": check_result.latency_ms,
            "details": check_result.details or {},
            "source": "veris-sentinel",
            "environment": os.getenv('ENVIRONMENT', 'production')
        }
        
        # Repository dispatch payload
        payload = {
            "event_type": "sentinel-alert",
            "client_payload": client_payload
        }
        
        return payload
    
    async def _send_repository_dispatch(self, payload: Dict[str, Any]) -> bool:
        """
        Send repository dispatch event to GitHub.
        
        Args:
            payload: The payload to send
            
        Returns:
            bool: True if successful
        """
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Veris-Sentinel-Webhook/1.0"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.dispatch_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 204:
                        logger.info("Repository dispatch sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"GitHub API error {response.status}: {error_text}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.error("Timeout sending repository dispatch")
            return False
        except Exception as e:
            logger.error(f"Error sending repository dispatch: {str(e)}")
            return False
    
    def _record_alert_sent(self, check_result: CheckResult):
        """Record that an alert was sent for rate limiting."""
        self.recent_alerts.append(datetime.now().timestamp())
    
    def _generate_signature(self, payload: str) -> str:
        """
        Generate HMAC signature for webhook security.
        
        Args:
            payload: The payload to sign
            
        Returns:
            str: The HMAC signature
        """
        if not self.webhook_secret:
            return ""
        
        signature = hmac.new(
            self.webhook_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    async def test_connectivity(self) -> bool:
        """
        Test connectivity to GitHub API.
        
        Returns:
            bool: True if GitHub API is accessible
        """
        if not self.github_token:
            logger.error("GitHub token not configured")
            return False
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.github_api_base}/repos/{self.github_repo}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    
                    if response.status == 200:
                        logger.info("GitHub API connectivity test successful")
                        return True
                    else:
                        logger.error(f"GitHub API test failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"GitHub API connectivity test failed: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of GitHub webhook alerter.
        
        Returns:
            Dict containing status information
        """
        return {
            "enabled": bool(self.github_token),
            "github_repo": self.github_repo,
            "rate_limit_remaining": max(0, self.max_alerts_per_minute - len(self.recent_alerts)),
            "recent_alerts_count": len(self.recent_alerts),
            "last_alert_time": max(self.recent_alerts) if self.recent_alerts else None
        }


# Example usage and testing
async def main():
    """Test the GitHub webhook alerter."""
    alerter = GitHubWebhookAlerter()
    
    # Test connectivity
    if await alerter.test_connectivity():
        print("✅ GitHub API connectivity successful")
    else:
        print("❌ GitHub API connectivity failed")
    
    # Test alert sending (example)
    test_result = CheckResult(
        check_id="test-alert",
        timestamp=datetime.now(),
        status="fail",
        latency_ms=100.0,
        message="Test alert for GitHub integration",
        details={"test": True}
    )
    
    success = await alerter.send_alert(test_result)
    print(f"Test alert sent: {success}")
    
    # Show status
    status = alerter.get_status()
    print(f"Alerter status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())