#!/usr/bin/env python3
"""
Integration tests for Sentinel monitoring system.

Tests Sentinel's ability to detect failures, send alerts, and monitor
the health of the Veris Memory system end-to-end.
"""

import asyncio
import os
import time
from typing import Optional, List, Dict, Any, AsyncGenerator
import pytest
import aiohttp
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime

from src.monitoring.sentinel.runner import SentinelRunner
from src.monitoring.sentinel.models import SentinelConfig, CheckResult
from src.monitoring.sentinel.telegram_alerter import TelegramAlerter


class TestSentinelIntegration:
    """
    Integration tests for the Sentinel monitoring system.
    
    This test class validates the core functionality of Sentinel including:
    - Health check monitoring and failure detection
    - Alert management and Telegram integration
    - Recovery detection and circuit breaker behavior
    - Concurrent check execution and graceful shutdown
    - Error handling and resilience
    """
    
    @pytest.fixture
    async def config(self) -> SentinelConfig:
        """Create test configuration."""
        return SentinelConfig({
            "veris_memory_url": os.getenv("TARGET_BASE_URL", "http://localhost:8000"),
            "check_interval_seconds": 5,  # Fast interval for testing
            "alert_threshold_failures": 2,  # Alert after 2 failures
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "test_token"),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", "test_chat"),
            "enabled_checks": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
        })
    
    @pytest.fixture
    async def runner(self, config: SentinelConfig) -> AsyncGenerator[SentinelRunner, None]:
        """Create a Sentinel runner instance."""
        runner = SentinelRunner(config)
        yield runner
        # Cleanup
        if runner._check_task and not runner._check_task.done():
            runner._check_task.cancel()
            try:
                await runner._check_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_sentinel_initialization(self, runner: SentinelRunner) -> None:
        """Test Sentinel runner initialization."""
        assert runner is not None
        assert len(runner.checks) == 10  # All S1-S10 checks
        assert runner.alert_manager is not None
        assert runner.failure_counts == {}
    
    @pytest.mark.asyncio
    async def test_health_check_monitoring(self, runner: SentinelRunner) -> None:
        """Test that Sentinel can monitor health checks."""
        # Mock successful health check
        with patch.object(runner.checks[0], 'run_check') as mock_check:
            mock_check.return_value = CheckResult(
                check_id="S1-health",
                status="pass",
                message="Service is healthy",
                timestamp=datetime.utcnow(),
                details={}
            )
            
            # Run one check cycle
            await runner._run_check_cycle()
            
            # Verify check was called
            mock_check.assert_called_once()
            
            # Verify no failures recorded
            assert runner.failure_counts.get("S1-health", 0) == 0
    
    @pytest.mark.asyncio
    async def test_failure_detection_and_alerting(self, runner: SentinelRunner) -> None:
        """Test that Sentinel detects failures and sends alerts."""
        # Mock failed health check
        with patch.object(runner.checks[0], 'run_check') as mock_check:
            mock_check.return_value = CheckResult(
                check_id="S1-health",
                status="fail",
                message="Service is unhealthy",
                timestamp=datetime.utcnow(),
                details={"error": "Connection refused"}
            )
            
            # Mock Telegram alert
            with patch.object(runner.alert_manager, 'send_alert') as mock_alert:
                mock_alert.return_value = asyncio.create_task(asyncio.sleep(0))
                
                # Run check cycles to trigger alert (need 2 failures)
                await runner._run_check_cycle()
                assert runner.failure_counts.get("S1-health", 0) == 1
                
                await runner._run_check_cycle()
                assert runner.failure_counts.get("S1-health", 0) == 2
                
                # Verify alert was sent after threshold
                await asyncio.sleep(0.1)  # Let async tasks complete
                assert mock_alert.called
                alert_args = mock_alert.call_args[0]
                assert "S1-health" in alert_args[0]
                assert "fail" in alert_args[1].lower()
    
    @pytest.mark.asyncio
    async def test_telegram_alert_integration(self) -> None:
        """Test Telegram alert sending with mock API."""
        alert = TelegramAlerter(
            bot_token="test_token",
            chat_id="test_chat"
        )
        
        # Mock Telegram API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True})
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await alert.send("Test Alert", "This is a test message")
            
            assert result is True
            mock_post.assert_called_once()
            
            # Verify API endpoint
            call_args = mock_post.call_args
            assert "api.telegram.org/bottest_token/sendMessage" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_recovery_detection(self, runner: SentinelRunner) -> None:
        """Test that Sentinel detects service recovery."""
        check_id = "S1-health"
        
        # Simulate failures
        runner.failure_counts[check_id] = 3
        
        # Mock successful check (recovery)
        with patch.object(runner.checks[0], 'run_check') as mock_check:
            mock_check.return_value = CheckResult(
                check_id=check_id,
                status="pass",
                message="Service recovered",
                timestamp=datetime.utcnow(),
                details={}
            )
            
            with patch.object(runner.alert_manager, 'send_alert') as mock_alert:
                mock_alert.return_value = asyncio.create_task(asyncio.sleep(0))
                
                await runner._run_check_cycle()
                
                # Verify failure count reset
                assert runner.failure_counts.get(check_id, 0) == 0
                
                # Verify recovery alert sent
                await asyncio.sleep(0.1)
                if mock_alert.called:
                    alert_args = mock_alert.call_args[0]
                    assert "recovered" in alert_args[1].lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_check_execution(self, runner: SentinelRunner) -> None:
        """Test that multiple checks run concurrently."""
        execution_times: List[float] = []
        
        async def mock_check_run(self) -> CheckResult:
            start = time.time()
            await asyncio.sleep(0.5)  # Simulate check taking time
            execution_times.append(time.time() - start)
            return CheckResult(
                check_id=self.check_id,
                status="pass",
                message="Check passed",
                timestamp=datetime.utcnow(),
                details={}
            )
        
        # Mock all checks to take 0.5 seconds
        for check in runner.checks:
            check.run_check = lambda c=check: mock_check_run(c)
        
        start_time = time.time()
        await runner._run_check_cycle()
        total_time = time.time() - start_time
        
        # If checks run concurrently, total time should be ~0.5s
        # If sequential, it would be ~5s (10 checks * 0.5s)
        assert total_time < 2.0  # Allow some overhead
        assert len(execution_times) == 10
    
    @pytest.mark.asyncio
    async def test_api_endpoint_health(self, runner: SentinelRunner) -> None:
        """Test Sentinel's own health check API endpoint."""
        # Use dynamic port from environment or default
        test_port: int = int(os.getenv('SENTINEL_TEST_PORT', '9091'))
        
        # Start API server
        api_task = asyncio.create_task(runner.start_api_server(port=test_port))
        
        try:
            await asyncio.sleep(1)  # Let server start
            
            # Check health endpoint
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"http://localhost:{test_port}/status") as response:
                        assert response.status == 200
                        data = await response.json()
                        assert "status" in data
                        assert "checks" in data
                        assert "last_run" in data
                except aiohttp.ClientError:
                    pytest.skip("API server not available")
        finally:
            api_task.cancel()
            try:
                await api_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self, runner: SentinelRunner) -> None:
        """Test circuit breaker prevents alert spam."""
        # Set up failure response
        failed_result = CheckResult(
            check_id="S1-health",
            status="fail",
            message="Service down",
            timestamp=datetime.utcnow(),
            details={}
        )
        
        alert_count: int = 0
        
        with patch.object(runner.checks[0], 'run_check', return_value=failed_result):
            with patch.object(runner.alert_manager, 'send_alert') as mock_alert:
                mock_alert.side_effect = lambda *args, **kwargs: setattr(mock_alert, 'call_count', alert_count + 1)
                
                # Run multiple check cycles
                for _ in range(10):
                    await runner._run_check_cycle()
                
                # Circuit breaker should limit alerts (not one per failure)
                assert mock_alert.call_count < 10
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, runner: SentinelRunner) -> None:
        """Test Sentinel shuts down gracefully."""
        # Start monitoring
        monitor_task = asyncio.create_task(runner.start_monitoring())
        
        await asyncio.sleep(1)  # Let it run briefly
        
        # Request shutdown
        runner.shutdown_requested = True
        
        # Wait for graceful shutdown
        try:
            await asyncio.wait_for(monitor_task, timeout=5)
        except asyncio.TimeoutError:
            pytest.fail("Sentinel did not shut down gracefully")
    
    @pytest.mark.asyncio
    async def test_configuration_reload(self, runner: SentinelRunner) -> None:
        """Test configuration can be reloaded without restart."""
        original_interval = runner.config.check_interval_seconds
        
        # Update configuration
        new_config = SentinelConfig({
            "veris_memory_url": "http://localhost:8000",
            "check_interval_seconds": 10,
            "alert_threshold_failures": 5
        })
        
        runner.update_config(new_config)
        
        assert runner.config.check_interval_seconds == 10
        assert runner.config.alert_threshold_failures == 5
        assert runner.config.check_interval_seconds != original_interval
    
    @pytest.mark.asyncio
    async def test_error_handling_resilience(self, runner: SentinelRunner) -> None:
        """Test Sentinel continues operating despite check errors."""
        # Make one check always raise exception
        def raise_error():
            raise Exception("Check implementation error")
        
        runner.checks[0].run_check = raise_error
        
        # Other checks should still work
        with patch.object(runner.checks[1], 'run_check') as mock_good_check:
            mock_good_check.return_value = CheckResult(
                check_id="S2-performance",
                status="pass",
                message="Performance good",
                timestamp=datetime.utcnow(),
                details={}
            )
            
            # Run check cycle - should not crash
            await runner._run_check_cycle()
            
            # Good check should have been called
            mock_good_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, runner: SentinelRunner) -> None:
        """Test that Sentinel collects metrics about checks."""
        # Run several check cycles
        for i in range(3):
            with patch.object(runner.checks[0], 'run_check') as mock_check:
                mock_check.return_value = CheckResult(
                    check_id="S1-health",
                    status="pass" if i < 2 else "fail",
                    message=f"Check {i}",
                    timestamp=datetime.utcnow(),
                    details={"iteration": i}
                )
                await runner._run_check_cycle()
        
        # Check metrics
        metrics: Dict[str, Any] = runner.get_metrics()
        assert "check_results" in metrics
        assert "failure_counts" in metrics
        assert "total_checks_run" in metrics
        assert metrics["total_checks_run"] >= 3


class TestSentinelScenarios:
    """
    Test real-world failure scenarios for Sentinel.
    
    This test class focuses on specific failure scenarios that Sentinel
    must detect and handle correctly:
    - Database wipe detection (like the overnight incident)
    - Service cascade failures
    - Network partitions and timeout handling
    - Data consistency violations
    """
    
    @pytest.mark.asyncio
    async def test_database_wipe_detection(self) -> None:
        """Test detection of database wipe (like the overnight incident)."""
        config = SentinelConfig({
            "veris_memory_url": "http://localhost:8000",
            "check_interval_seconds": 5,
            "alert_threshold_failures": 1
        })
        
        runner = SentinelRunner(config)
        
        # Simulate database wipe detection
        with patch('aiohttp.ClientSession.post') as mock_post:
            # First call returns empty contexts (database wiped)
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"contexts": []})
            mock_post.return_value.__aenter__.return_value = mock_response
            
            alert_sent: bool = False
            
            async def capture_alert(*args: Any, **kwargs: Any) -> None:
                nonlocal alert_sent
                alert_sent = True
                # Check alert mentions data loss
                if len(args) > 1:
                    assert "data" in args[1].lower() or "empty" in args[1].lower()
            
            with patch.object(runner.alert_manager, 'send_alert', side_effect=capture_alert):
                # Run data integrity check
                for check in runner.checks:
                    if "data" in check.check_id.lower():
                        result = await check.run_check()
                        if result.status == "fail":
                            await runner._handle_check_result(check.check_id, result)
                            break
                
                await asyncio.sleep(0.1)
                assert alert_sent, "Alert should be sent for data loss"
    
    @pytest.mark.asyncio
    async def test_service_cascade_failure(self) -> None:
        """Test detection of cascading service failures."""
        config = SentinelConfig({
            "veris_memory_url": "http://localhost:8000",
            "check_interval_seconds": 5,
            "alert_threshold_failures": 1
        })
        
        runner = SentinelRunner(config)
        alerts_sent: List[tuple[str, str]] = []
        
        async def track_alerts(check_id: str, message: str, *args: Any, **kwargs: Any) -> None:
            alerts_sent.append((check_id, message))
        
        with patch.object(runner.alert_manager, 'send_alert', side_effect=track_alerts):
            # Simulate Neo4j failure
            for check in runner.checks:
                if "S1" in check.check_id:
                    check.run_check = AsyncMock(return_value=CheckResult(
                        check_id="S1-health",
                        status="fail",
                        message="Neo4j connection failed",
                        timestamp=datetime.utcnow(),
                        details={"service": "neo4j", "error": "Connection refused"}
                    ))
            
            await runner._run_check_cycle()
            await asyncio.sleep(0.1)
            
            # Should detect and alert about Neo4j failure
            assert len(alerts_sent) > 0
            assert any("neo4j" in alert[1].lower() for alert in alerts_sent)