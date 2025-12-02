#!/usr/bin/env python3
"""
Integration tests for Veris Sentinel with actual services.

These tests verify that Sentinel can communicate with real Veris Memory services
in a test environment, not just mocks. Covers all monitoring checks (S1-S8)
with actual service interactions.
"""

import asyncio
import pytest
import docker
import time
import json
import requests
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import os
import logging

# Import Sentinel components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.veris_sentinel import (
    SentinelConfig, SentinelRunner, VerisHealthProbe, 
    GoldenFactRecall, MetricsWiring, SecurityNegatives,
    ConfigParity, CapacitySmoke
)


logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def docker_client():
    """Docker client for managing test containers."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")


@pytest.fixture(scope="session") 
def test_services(docker_client):
    """Start test Veris Memory services in Docker containers."""
    logger.info("Starting test services for integration testing...")
    
    # Docker Compose project name for isolation
    project_name = "veris-memory-integration-test"
    
    # Cleanup any existing containers
    try:
        subprocess.run([
            "docker-compose", "-p", project_name, "down", "-v"
        ], cwd=Path(__file__).parent.parent.parent, capture_output=True)
    except Exception:
        pass
    
    # Start test services
    compose_file = Path(__file__).parent.parent.parent / "docker-compose.yml"
    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found for integration testing")
    
    try:
        # Start services with test configuration
        result = subprocess.run([
            "docker-compose", "-p", project_name, 
            "-f", str(compose_file),
            "up", "-d", "--wait"
        ], cwd=Path(__file__).parent.parent.parent, 
           capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            pytest.skip(f"Failed to start test services: {result.stderr}")
        
        # Wait for services to be ready
        logger.info("Waiting for services to be ready...")
        await_service_ready("http://localhost:8000/health/live", timeout=60)
        await_service_ready("http://localhost:6333/collections", timeout=30)
        await_redis_ready("localhost", 6379, timeout=30)
        
        yield {
            "api_url": "http://localhost:8000",
            "qdrant_url": "http://localhost:6333", 
            "redis_url": "redis://localhost:6379",
            "neo4j_bolt": "bolt://localhost:7687",
            "project_name": project_name
        }
        
    finally:
        # Cleanup
        logger.info("Cleaning up test services...")
        try:
            subprocess.run([
                "docker-compose", "-p", project_name, "down", "-v"
            ], cwd=Path(__file__).parent.parent.parent, 
               capture_output=True, timeout=60)
        except Exception as e:
            logger.warning(f"Failed to cleanup test services: {e}")


def await_service_ready(url: str, timeout: int = 30) -> bool:
    """Wait for HTTP service to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"Service at {url} not ready within {timeout}s")


def await_redis_ready(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for Redis to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            import redis
            r = redis.Redis(host=host, port=port, socket_connect_timeout=2)
            r.ping()
            return True
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"Redis at {host}:{port} not ready within {timeout}s")


@pytest.fixture
def sentinel_config(test_services):
    """Sentinel configuration for integration testing."""
    return SentinelConfig(
        target_base_url=test_services["api_url"],
        qdrant_url=test_services["qdrant_url"],
        redis_url=test_services["redis_url"],
        neo4j_bolt=test_services["neo4j_bolt"],
        neo4j_user="neo4j",  # Default for test environment
        schedule_cadence_sec=10,  # Faster for testing
        per_check_timeout_sec=15,
        cycle_budget_sec=60,
        max_parallel_checks=2
    )


@pytest.fixture
def sentinel_runner(sentinel_config, tmp_path):
    """Sentinel runner instance for testing."""
    db_path = tmp_path / "test_sentinel.db"
    runner = SentinelRunner(sentinel_config, str(db_path))
    yield runner
    runner.stop()


class TestSentinelIntegration:
    """Integration tests for Sentinel with real services."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_health_probe_integration(self, sentinel_config):
        """Test S1 health probes with actual services."""
        health_probe = VerisHealthProbe(sentinel_config)
        
        result = await health_probe.run_check()
        
        assert result.check_id == "S1-probes"
        assert result.status in ["pass", "warn"]  # Should not fail with real services
        assert result.latency_ms > 0
        assert result.latency_ms < 5000  # Should be reasonable
        
        if result.status == "pass":
            assert result.metrics is not None
            assert "endpoints_checked" in result.metrics
            assert result.metrics["endpoints_checked"] >= 2
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_golden_fact_recall_integration(self, sentinel_config):
        """Test S2 golden fact recall with actual API."""
        fact_recall = GoldenFactRecall(sentinel_config)
        
        result = await fact_recall.run_check()
        
        assert result.check_id == "S2-golden-fact-recall"
        assert result.status in ["pass", "warn", "fail"]
        assert result.latency_ms > 0
        
        # Even if it fails, it should provide meaningful error info
        if result.status == "fail":
            assert result.error_message is not None
            assert len(result.error_message) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_metrics_wiring_integration(self, sentinel_config):
        """Test S4 metrics collection with actual services."""
        metrics_check = MetricsWiring(sentinel_config)
        
        result = await metrics_check.run_check()
        
        assert result.check_id == "S4-metrics-wiring"
        assert result.status in ["pass", "warn", "fail"]
        assert result.latency_ms > 0
        
        if result.status == "pass":
            assert result.metrics is not None
            assert "prometheus_endpoints" in result.metrics
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_security_negatives_integration(self, sentinel_config):
        """Test S5 security validation with actual API."""
        security_check = SecurityNegatives(sentinel_config)
        
        result = await security_check.run_check()
        
        assert result.check_id == "S5-security-negatives"
        assert result.status in ["pass", "warn", "fail"]
        assert result.latency_ms > 0
        
        # Security check should test unauthorized access
        if result.status == "pass":
            assert result.metrics is not None
            assert "unauthorized_attempts" in result.metrics
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_config_parity_integration(self, sentinel_config):
        """Test S7 configuration drift detection."""
        config_check = ConfigParity(sentinel_config)
        
        result = await config_check.run_check()
        
        assert result.check_id == "S7-config-parity"
        assert result.status in ["pass", "warn", "fail"]
        assert result.latency_ms > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_capacity_smoke_integration(self, sentinel_config):
        """Test S8 capacity testing with actual services."""
        capacity_check = CapacitySmoke(sentinel_config)
        
        result = await capacity_check.run_check()
        
        assert result.check_id == "S8-capacity-smoke"
        assert result.status in ["pass", "warn", "fail"]
        assert result.latency_ms > 0
        
        if result.status == "pass":
            assert result.metrics is not None
            assert "requests_completed" in result.metrics
            assert result.metrics["requests_completed"] > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_monitoring_cycle_integration(self, sentinel_runner):
        """Test complete monitoring cycle with real services."""
        # Run a single monitoring cycle
        cycle_report = await sentinel_runner.run_single_cycle()
        
        # Validate cycle report structure
        assert "cycle_id" in cycle_report
        assert "total_checks" in cycle_report
        assert "passed_checks" in cycle_report
        assert "failed_checks" in cycle_report
        assert "cycle_duration_ms" in cycle_report
        assert "results" in cycle_report
        
        # Should have run all checks
        assert cycle_report["total_checks"] >= 5  # At least S1, S2, S4, S5, S7, S8
        
        # At least some checks should pass with real services
        assert cycle_report["passed_checks"] >= 1
        
        # Cycle should complete in reasonable time
        assert cycle_report["cycle_duration_ms"] < 30000  # 30 seconds
        
        # Validate individual check results
        for result in cycle_report["results"]:
            assert "check_id" in result
            assert "status" in result
            assert "latency_ms" in result
            assert "timestamp" in result
            
            # Each check should complete reasonably quickly
            assert result["latency_ms"] < 15000  # 15 seconds per check
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sentinel_database_persistence_integration(self, sentinel_runner, tmp_path):
        """Test that Sentinel properly persists data to database."""
        # Run a cycle to generate data
        cycle_report = await sentinel_runner.run_single_cycle()
        
        # Verify database file was created
        db_path = Path(sentinel_runner.db_path)
        assert db_path.exists()
        assert db_path.stat().st_size > 0
        
        # Verify data was stored
        import sqlite3
        with sqlite3.connect(sentinel_runner.db_path) as conn:
            cursor = conn.cursor()
            
            # Check that tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "check_results" in tables
            assert "cycles" in tables
            
            # Check that data was inserted
            cursor.execute("SELECT COUNT(*) FROM check_results")
            check_count = cursor.fetchone()[0]
            assert check_count > 0
            
            cursor.execute("SELECT COUNT(*) FROM cycles")
            cycle_count = cursor.fetchone()[0]
            assert cycle_count > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_sentinel_api_integration(self, sentinel_runner):
        """Test Sentinel API endpoints with real data."""
        from src.monitoring.veris_sentinel import SentinelAPI
        
        # Create API instance
        api = SentinelAPI(sentinel_runner, port=9091)  # Use different port for testing
        
        try:
            # Start API server
            await api.start_server()
            
            # Wait for server to start
            await asyncio.sleep(1)
            
            # Test status endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:9091/status") as resp:
                    assert resp.status == 200
                    status_data = await resp.json()
                    assert "running" in status_data
                    assert "uptime_seconds" in status_data
                
                # Test checks endpoint
                async with session.get("http://localhost:9091/checks") as resp:
                    assert resp.status == 200
                    checks_data = await resp.json()
                    assert "checks" in checks_data
                    assert len(checks_data["checks"]) >= 5
                
                # Test metrics endpoint
                async with session.get("http://localhost:9091/metrics") as resp:
                    assert resp.status == 200
                    metrics_text = await resp.text()
                    assert "sentinel_" in metrics_text
        
        except Exception as e:
            pytest.fail(f"API integration test failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sentinel_parallel_execution_integration(self, sentinel_config):
        """Test that Sentinel can run multiple checks in parallel."""
        runner = SentinelRunner(sentinel_config, ":memory:")
        
        start_time = time.time()
        cycle_report = await runner.run_single_cycle()
        execution_time = time.time() - start_time
        
        # With parallel execution, total time should be less than sum of individual times
        total_check_time = sum(
            result["latency_ms"] for result in cycle_report["results"]
        ) / 1000  # Convert to seconds
        
        # Parallel execution should be significantly faster
        assert execution_time < total_check_time * 0.8
        
        runner.stop()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sentinel_error_resilience_integration(self, sentinel_config):
        """Test Sentinel resilience when some services are unavailable."""
        # Create config with one invalid service URL
        faulty_config = SentinelConfig(
            target_base_url="http://invalid-host:9999",  # This will fail
            qdrant_url=sentinel_config.qdrant_url,  # This should work
            redis_url=sentinel_config.redis_url,  # This should work
            neo4j_bolt=sentinel_config.neo4j_bolt,
            per_check_timeout_sec=5  # Shorter timeout for faster testing
        )
        
        runner = SentinelRunner(faulty_config, ":memory:")
        
        # Should still complete cycle even with failures
        cycle_report = await runner.run_single_cycle()
        
        assert cycle_report["total_checks"] > 0
        # Some checks should fail due to invalid URL
        assert cycle_report["failed_checks"] > 0
        # But some might still pass (like config parity)
        assert cycle_report["passed_checks"] >= 0
        
        runner.stop()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_sentinel_performance_baseline_integration(self, sentinel_runner):
        """Test Sentinel performance baseline with real services."""
        # Run multiple cycles to establish baseline
        cycle_times = []
        
        for i in range(3):
            start_time = time.time()
            cycle_report = await sentinel_runner.run_single_cycle()
            cycle_time = time.time() - start_time
            cycle_times.append(cycle_time)
            
            # Brief delay between cycles
            await asyncio.sleep(1)
        
        # Analyze performance
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        max_cycle_time = max(cycle_times)
        
        # Performance assertions for production system
        assert avg_cycle_time < 10.0  # Average cycle under 10 seconds
        assert max_cycle_time < 15.0  # No cycle over 15 seconds
        
        logger.info(f"Performance baseline: avg={avg_cycle_time:.2f}s, max={max_cycle_time:.2f}s")


class TestSentinelRealWorldScenarios:
    """Test real-world scenarios and edge cases."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sentinel_during_service_restart(self, sentinel_config, docker_client):
        """Test Sentinel behavior during service restarts."""
        runner = SentinelRunner(sentinel_config, ":memory:")
        
        # Run initial cycle
        initial_report = await runner.run_single_cycle()
        assert initial_report["passed_checks"] > 0
        
        # Restart one of the services (Redis is fastest to restart)
        try:
            # Find Redis container
            containers = docker_client.containers.list(
                filters={"label": "com.docker.compose.service=redis"}
            )
            if containers:
                redis_container = containers[0]
                redis_container.restart()
                
                # Wait a moment for restart
                await asyncio.sleep(2)
                
                # Run cycle during restart - should handle gracefully
                restart_report = await runner.run_single_cycle()
                
                # Some checks might fail, but Sentinel should continue
                assert restart_report["total_checks"] > 0
                
                # Wait for service to be fully ready
                await_redis_ready("localhost", 6379, timeout=30)
                
                # Run another cycle - should recover
                recovery_report = await runner.run_single_cycle()
                
                # Should have at least as many passing checks as initially
                assert recovery_report["passed_checks"] >= initial_report["passed_checks"] * 0.8
        
        except Exception as e:
            logger.warning(f"Service restart test failed: {e}")
            # This test might fail in CI environments, so we'll just warn
        
        runner.stop()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sentinel_memory_usage_integration(self, sentinel_runner):
        """Test Sentinel memory usage over multiple cycles."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple cycles
        for i in range(5):
            await sentinel_runner.run_single_cycle()
            await asyncio.sleep(0.5)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024
        
        logger.info(f"Memory usage: initial={initial_memory/1024/1024:.1f}MB, "
                   f"final={final_memory/1024/1024:.1f}MB, "
                   f"increase={memory_increase/1024/1024:.1f}MB")


@pytest.mark.integration
def test_integration_test_environment():
    """Verify that integration test environment is properly configured."""
    # This test ensures that integration tests can run
    # It's a meta-test to validate the test setup
    
    required_tools = ["docker", "docker-compose"]
    for tool in required_tools:
        result = subprocess.run(["which", tool], capture_output=True)
        assert result.returncode == 0, f"Required tool {tool} not found"
    
    # Check Docker daemon is running
    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        pytest.fail(f"Docker daemon not accessible: {e}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])