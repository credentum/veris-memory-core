#!/usr/bin/env python3
"""
Sentinel Runner - Main orchestration and scheduling logic.

This module contains the core SentinelRunner class that manages
check execution, scheduling, data persistence, and reporting.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .models import CheckResult, SentinelConfig
from .checks import CHECK_REGISTRY
from .alert_manager import AlertManager
from src.core.import_utils import safe_import

logger = logging.getLogger(__name__)


class SentinelRunner:
    """Main Veris Sentinel runner with scheduling and reporting."""

    # PR #390: Per-check intervals to reduce system load
    # PR #397: Optimized frequencies - CI/CD-only checks disabled at runtime
    #
    # Check Categories:
    # - Runtime (continuous): S1, S11 - critical health/security monitoring
    # - Spot-check (hourly): S2, S5, S6, S10 - periodic validation
    # - CI/CD only (interval=0): S3, S4, S7, S8, S9 - static behavior, run on deploy
    #
    # Values are in seconds. Interval=0 means CI/CD only (disabled at runtime).
    # Can be overridden via environment variables: SENTINEL_INTERVAL_<CHECK_NUM>
    DEFAULT_CHECK_INTERVALS = {
        "S1-probes": 60,              # Health probes - every 1 minute (critical)
        "S2-golden-fact-recall": 3600, # Golden fact - hourly spot-check
        "S3-paraphrase-robustness": 0, # CI/CD only - static behavior, expensive
        "S4-metrics-wiring": 0,        # CI/CD only - only changes on deploy
        "S5-security-negatives": 3600, # Security - hourly spot-check
        "S6-backup-restore": 21600,    # Backup - every 6 hours (filesystem scan)
        "S7-config-parity": 0,         # CI/CD only - config changes on deploy
        "S8-capacity-smoke": 0,        # CI/CD only - performance baseline
        "S9-graph-intent": 0,          # CI/CD only - schema doesn't change
        "S10-content-pipeline": 3600,  # Content - hourly spot-check
        "S11-firewall-status": 300,    # Firewall - every 5 minutes (security)
    }

    def __init__(self, config: SentinelConfig, db_path: str = "/var/lib/sentinel/sentinel.db") -> None:
        self.config = config
        self.db_path = db_path
        self.running = False

        # Initialize checks based on configuration
        self.checks = self._initialize_checks()

        # Initialize alert manager
        self.alert_manager = self._initialize_alert_manager()

        # Ring buffers for data retention
        self.failures: deque = deque(maxlen=200)
        self.reports: deque = deque(maxlen=50)
        self.traces: deque = deque(maxlen=500)

        # External service resilience tracking
        self.webhook_failures = 0
        self.github_failures = 0

        # Statistics tracking
        self.check_statistics = defaultdict(lambda: {
            'total_runs': 0,
            'successes': 0,
            'failures': 0,
            'warnings': 0,
            'total_time_ms': 0.0,
            'last_result': None
        })

        # PR #390: Track last run time for per-check intervals
        self.last_check_time: Dict[str, float] = {}
        self.check_intervals = self._load_check_intervals()
        
        # PR #247: Validate critical environment variables at startup
        self._validate_environment()

        # Initialize database
        self._init_database()

    def _validate_environment(self) -> None:
        """Validate critical environment variables at startup.

        Set SENTINEL_FAIL_FAST=true to exit on validation failures.
        """
        import os
        import sys

        # Check if fail-fast mode is enabled
        fail_fast = os.getenv("SENTINEL_FAIL_FAST", "false").lower() == "true"

        # Critical environment variables required for S7 config parity check
        critical_vars = ["LOG_LEVEL", "ENVIRONMENT"]
        missing_vars = []
        invalid_vars = []

        for var_name in critical_vars:
            value = os.getenv(var_name)
            # Check for None, empty string, or whitespace-only values
            if not value or value.strip() == "":
                missing_vars.append(var_name)

        # Validate LOG_LEVEL if set
        log_level = os.getenv("LOG_LEVEL", "").upper()
        if log_level and log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            invalid_vars.append(f"LOG_LEVEL='{log_level}' (expected: DEBUG, INFO, WARNING, ERROR, CRITICAL)")

        # Validate ENVIRONMENT if set
        environment = os.getenv("ENVIRONMENT", "").lower()
        if environment and environment not in ["development", "staging", "production", "test"]:
            invalid_vars.append(f"ENVIRONMENT='{environment}' (expected: development, staging, production, test)")

        # Report validation results
        has_errors = bool(missing_vars or invalid_vars)

        if missing_vars:
            msg = f"Missing or empty critical environment variables: {', '.join(missing_vars)}"
            if fail_fast:
                logger.error(msg)
            else:
                logger.warning(msg)

        if invalid_vars:
            msg = f"Invalid environment variable values: {', '.join(invalid_vars)}"
            if fail_fast:
                logger.error(msg)
            else:
                logger.warning(msg)

        if has_errors:
            if fail_fast:
                logger.error("SENTINEL_FAIL_FAST is enabled - exiting due to environment validation failures")
                sys.exit(1)
            else:
                logger.warning("S7 config parity check may fail (set SENTINEL_FAIL_FAST=true to exit on validation failures)")

        logger.info(f"Environment validation complete. LOG_LEVEL={log_level or 'NOT_SET'}, ENVIRONMENT={environment or 'NOT_SET'}")

    def _initialize_alert_manager(self) -> Optional[AlertManager]:
        """Initialize alert manager if configured."""
        import os
        
        # Check if Telegram is configured
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        # Check if GitHub is configured
        github_token = os.getenv("GITHUB_TOKEN")
        github_repo = os.getenv("GITHUB_REPO", "credentum/veris-memory")
        
        # Get configuration parameters
        dedup_window = int(os.getenv("ALERT_DEDUP_WINDOW_MIN", "30"))
        alert_threshold = int(os.getenv("ALERT_THRESHOLD_FAILURES", "3"))
        
        # Always create AlertManager - it will handle missing/invalid services gracefully
        try:
            return AlertManager(
                telegram_token=telegram_token,
                telegram_chat_id=telegram_chat_id,
                github_token=github_token,
                github_repo=github_repo,
                dedup_window_minutes=dedup_window,
                alert_threshold_failures=alert_threshold
            )
        except Exception as e:
            logger.warning(f"Alert manager initialization failed: {e}")
            logger.warning("Alerts will be logged only")
            return None
    
    def _initialize_checks(self) -> Dict[str, Any]:
        """Initialize check instances based on configuration."""
        checks = {}
        
        for check_id, check_class in CHECK_REGISTRY.items():
            if self.config.is_check_enabled(check_id):
                try:
                    checks[check_id] = check_class(self.config)
                    logger.info(f"Initialized check {check_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize check {check_id}: {e}")
        
        logger.info(f"Initialized {len(checks)} checks out of {len(CHECK_REGISTRY)} available")
        return checks

    def _load_check_intervals(self) -> Dict[str, int]:
        """Load check intervals from environment or use defaults.

        PR #390: Per-check intervals to reduce system load.

        Environment variable format:
            SENTINEL_INTERVAL_S1=120  # Override S1-probes to 2 minutes
            SENTINEL_INTERVAL_S6=3600 # Override S6-backup-restore to 1 hour

        Returns:
            Dictionary mapping check_id to interval in seconds
        """
        intervals = dict(self.DEFAULT_CHECK_INTERVALS)

        # Allow per-check interval overrides via environment variables
        for check_id in CHECK_REGISTRY.keys():
            # Extract check number (e.g., "S1" from "S1-probes")
            check_num = check_id.split("-")[0]
            env_var = f"SENTINEL_INTERVAL_{check_num}"
            env_value = os.getenv(env_var)

            if env_value:
                try:
                    intervals[check_id] = int(env_value)
                    logger.info(f"Check interval override: {check_id} = {env_value}s (from {env_var})")
                except ValueError:
                    logger.warning(f"Invalid interval value for {env_var}: {env_value}")

        # Log configured intervals
        logger.info("Per-check intervals configured:")
        for check_id, interval in sorted(intervals.items()):
            if interval == 0:
                human_interval = "CI/CD only"
            elif interval >= 3600:
                human_interval = f"{interval // 3600}h"
            elif interval >= 60:
                human_interval = f"{interval // 60}m"
            else:
                human_interval = f"{interval}s"
            logger.info(f"  {check_id}: {human_interval}")

        return intervals

    def _is_check_due(self, check_id: str) -> bool:
        """Check if a check is due to run based on its interval.

        PR #390: Per-check intervals support.
        PR #397: CI/CD-only checks (interval=0) are never due at runtime.

        Returns:
            True if the check should run, False if it should be skipped
        """
        interval = self.check_intervals.get(check_id, 60)  # Default 60s

        # PR #397: Interval of 0 means CI/CD only - never run at runtime
        if interval == 0:
            logger.debug(f"Skipping {check_id}: CI/CD-only check (interval=0)")
            return False

        now = time.time()
        last_run = self.last_check_time.get(check_id, 0)

        time_since_last = now - last_run
        is_due = time_since_last >= interval

        if not is_due:
            remaining = interval - time_since_last
            logger.debug(f"Skipping {check_id}: {remaining:.0f}s until next run")

        return is_due

    def _init_database(self) -> None:
        """Initialize SQLite database for persistent storage."""
        try:
            # Validate database path to prevent directory traversal
            db_path = Path(self.db_path).resolve()
            
            # Ensure path is within expected directory
            allowed_dirs = [
                Path("/var/lib/sentinel").resolve(),
                Path("/tmp").resolve(),
                Path.home().resolve() / ".sentinel"
            ]
            
            if not any(str(db_path).startswith(str(allowed)) for allowed in allowed_dirs):
                raise ValueError(f"Database path {db_path} is not in an allowed directory")
            
            # Ensure directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS check_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        check_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        status TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        message TEXT,
                        details TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alert_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        check_id TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        message TEXT,
                        timestamp TEXT NOT NULL,
                        resolved_at TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_check_results_timestamp 
                    ON check_results(timestamp)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_check_results_check_id 
                    ON check_results(check_id)
                ''')
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    async def start(self) -> None:
        """Start the sentinel monitoring loop."""
        if self.running:
            logger.warning("Sentinel is already running")
            return
        
        self.running = True
        logger.info("Starting Veris Sentinel monitoring")
        
        # Start periodic summary task
        summary_task = asyncio.create_task(self._periodic_summary_task())
        
        try:
            while self.running:
                await self._run_check_cycle()
                await asyncio.sleep(self.config.check_interval_seconds)
        except Exception as e:
            logger.error(f"Sentinel monitoring loop failed: {e}")
        finally:
            self.running = False
    
    async def stop(self) -> None:
        """Stop the sentinel monitoring."""
        self.running = False
        logger.info("Stopping Veris Sentinel monitoring")
    
    async def _periodic_summary_task(self) -> None:
        """Send periodic summaries via alert manager."""
        while self.running:
            try:
                # Wait for 24 hours (configurable)
                summary_interval = int(os.getenv("SUMMARY_INTERVAL_HOURS", "24"))
                await asyncio.sleep(summary_interval * 3600)
                
                if not self.running:
                    break
                
                # Collect results from the last period
                cutoff_time = datetime.utcnow() - timedelta(hours=summary_interval)
                check_results = []
                
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute('''
                            SELECT check_id, timestamp, status, latency_ms, message, details
                            FROM check_results
                            WHERE timestamp > ?
                            ORDER BY timestamp DESC
                        ''', (cutoff_time.isoformat(),))
                        
                        for row in cursor.fetchall():
                            result = CheckResult(
                                check_id=row[0],
                                timestamp=datetime.fromisoformat(row[1]),
                                status=row[2],
                                latency_ms=row[3],
                                message=row[4],
                                details=json.loads(row[5]) if row[5] else None
                            )
                            check_results.append(result)
                
                except Exception as e:
                    logger.error(f"Failed to collect summary data: {e}")
                    continue
                
                # Send summary via alert manager
                if self.alert_manager and check_results:
                    await self.alert_manager.send_summary(
                        period_hours=summary_interval,
                        check_results=check_results
                    )
                    logger.info(f"Sent {summary_interval}-hour summary")
                
            except Exception as e:
                logger.error(f"Error in periodic summary task: {e}")
    
    async def _run_check_cycle(self) -> None:
        """Execute one complete cycle of checks that are due.

        PR #390: Per-check intervals - only runs checks that are due based on
        their configured interval, reducing system load.
        """
        cycle_start = time.time()
        results = []

        # PR #390: Filter to only checks that are due
        due_checks = []
        skipped_checks = []
        for check_id, check_instance in self.checks.items():
            if check_instance.is_enabled():
                if self._is_check_due(check_id):
                    due_checks.append((check_id, check_instance))
                else:
                    skipped_checks.append(check_id)

        if skipped_checks:
            logger.debug(f"Skipped {len(skipped_checks)} checks (not due): {', '.join(skipped_checks)}")

        if not due_checks:
            logger.debug("No checks due this cycle")
            return

        logger.debug(f"Running {len(due_checks)} due checks: {', '.join(c[0] for c in due_checks)}")

        # Run due checks concurrently
        check_tasks = []
        for check_id, check_instance in due_checks:
            # Record run time BEFORE execution to prevent drift
            self.last_check_time[check_id] = time.time()
            task = asyncio.create_task(self._run_single_check(check_id, check_instance))
            check_tasks.append(task)

        if check_tasks:
            results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, CheckResult):
                await self._process_check_result(result)
            elif isinstance(result, Exception):
                logger.error(f"Check execution failed with exception: {result}")

        cycle_time = (time.time() - cycle_start) * 1000
        logger.info(f"Check cycle: {len(due_checks)} run, {len(skipped_checks)} skipped, {cycle_time:.1f}ms")
    
    async def _run_single_check(self, check_id: str, check_instance) -> CheckResult:
        """Run a single check with error handling."""
        try:
            result = await check_instance.execute()
            
            # Update statistics
            stats = self.check_statistics[check_id]
            stats['total_runs'] += 1
            stats['total_time_ms'] += result.latency_ms
            stats['last_result'] = result.to_dict()
            
            if result.status == 'pass':
                stats['successes'] += 1
            elif result.status == 'warn':
                stats['warnings'] += 1
            else:
                stats['failures'] += 1
            
            return result
            
        except Exception as e:
            # Create error result
            error_result = CheckResult(
                check_id=check_id,
                timestamp=datetime.utcnow(),
                status="fail",
                latency_ms=0.0,
                message=f"Check execution error: {str(e)}",
                details={"exception": str(e), "exception_type": type(e).__name__}
            )
            
            # Update statistics
            stats = self.check_statistics[check_id]
            stats['total_runs'] += 1
            stats['failures'] += 1
            stats['last_result'] = error_result.to_dict()
            
            return error_result
    
    async def _process_check_result(self, result: CheckResult) -> None:
        """Process a check result - store, alert, etc."""
        # Store in database
        await self._store_result(result)
        
        # Add to in-memory buffers
        if result.status in ['fail', 'warn']:
            self.failures.append(result.to_dict())
        
        self.traces.append({
            'timestamp': result.timestamp.isoformat(),
            'check_id': result.check_id,
            'status': result.status,
            'latency_ms': result.latency_ms
        })
        
        # Check for alerting conditions
        await self._check_alerting(result)
    
    async def _store_result(self, result: CheckResult) -> None:
        """Store check result in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO check_results 
                    (check_id, timestamp, status, latency_ms, message, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.check_id,
                    result.timestamp.isoformat(),
                    result.status,
                    result.latency_ms,
                    result.message,
                    json.dumps(result.details) if result.details else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store result: {e}")
    
    async def _check_alerting(self, result: CheckResult) -> None:
        """Check if alerting conditions are met."""
        if result.status != 'fail':
            return
        
        # Count recent failures for this check
        recent_failures = await self._count_recent_failures(result.check_id, minutes=5)
        
        if recent_failures >= self.config.alert_threshold_failures:
            await self._send_alert(result, recent_failures)
    
    async def _count_recent_failures(self, check_id: str, minutes: int = 5) -> int:
        """Count recent failures for a specific check."""
        try:
            cutoff_time = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM check_results
                    WHERE check_id = ? AND status = 'fail' AND timestamp > ?
                ''', (check_id, cutoff_time))
                
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count recent failures: {e}")
            return 0
    
    async def _send_alert(self, result: CheckResult, failure_count: int) -> None:
        """Send alert for check failure."""
        alert_message = f"ALERT: Check {result.check_id} has failed {failure_count} times in the last 5 minutes. Latest: {result.message}"
        
        logger.warning(alert_message)
        
        # Store alert in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO alert_history (check_id, alert_type, message, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (result.check_id, 'failure_threshold', alert_message, result.timestamp.isoformat()))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
        
        # Send external alerts (webhook, GitHub, etc.)
        await self._send_external_alerts(alert_message, result)
    
    async def _send_external_alerts(self, message: str, result: CheckResult) -> None:
        """Send alerts to external services via alert manager."""
        if self.alert_manager:
            try:
                await self.alert_manager.process_check_result(result)
            except Exception as e:
                logger.error(f"Failed to send external alert: {e}")
        else:
            # Fallback to logging if alert manager not configured
            logger.info(f"External alert (no manager configured): {message}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get overall status summary."""
        total_checks = len(self.checks)
        enabled_checks = sum(1 for check in self.checks.values() if check.is_enabled())
        
        recent_failures = len([f for f in list(self.failures)[-10:] if f])
        
        return {
            'running': self.running,
            'total_checks': total_checks,
            'enabled_checks': enabled_checks,
            'recent_failures': recent_failures,
            'check_statistics': dict(self.check_statistics),
            'last_cycle_time': datetime.utcnow().isoformat()
        }
    
    def get_check_history(self, check_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent history for a specific check."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT check_id, timestamp, status, latency_ms, message, details
                    FROM check_results
                    WHERE check_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (check_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        'check_id': row[0],
                        'timestamp': row[1],
                        'status': row[2],
                        'latency_ms': row[3],
                        'message': row[4],
                        'details': json.loads(row[5]) if row[5] else None
                    }
                    results.append(result)
                
                return results
        except Exception as e:
            logger.error(f"Failed to get check history: {e}")
            return []
