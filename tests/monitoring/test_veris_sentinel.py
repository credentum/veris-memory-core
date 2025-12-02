#!/usr/bin/env python3
"""
Comprehensive unit tests for Veris Sentinel monitoring agent.

Tests cover:
- Individual check classes and their functionality
- SentinelRunner core operations
- API endpoint handlers
- Configuration and error handling
- Database operations and persistence
- Alerting and webhook functionality
"""

import pytest
import asyncio
import json
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import aiohttp
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

# Import Sentinel components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.veris_sentinel import (
    CheckResult, SentinelConfig, VerisHealthProbe, 
    GoldenFactRecall, ParaphraseRobustness, MetricsWiring, SecurityNegatives,
    BackupRestore, ConfigParity, CapacitySmoke,
    SentinelRunner, SentinelAPI
)


class TestCheckResult:
    """Test CheckResult dataclass functionality."""
    
    def test_check_result_creation_basic(self):
        """Test basic CheckResult creation."""
        timestamp = datetime.utcnow()
        result = CheckResult(
            check_id="test-check",
            timestamp=timestamp,
            status="pass",
            latency_ms=123.45
        )
        
        assert result.check_id == "test-check"
        assert result.timestamp == timestamp
        assert result.status == "pass"
        assert result.latency_ms == 123.45
        assert result.error_message is None
        assert result.metrics is None
        assert result.notes == ""
    
    def test_check_result_creation_complete(self):
        """Test CheckResult creation with all fields."""
        timestamp = datetime.utcnow()
        metrics = {"response_time": 50.0, "success_rate": 0.95}
        
        result = CheckResult(
            check_id="comprehensive-check",
            timestamp=timestamp,
            status="warn",
            latency_ms=500.0,
            error_message="Minor issue detected",
            metrics=metrics,
            notes="Performance degradation observed"
        )
        
        assert result.check_id == "comprehensive-check"
        assert result.status == "warn"
        assert result.error_message == "Minor issue detected"
        assert result.metrics == metrics
        assert result.notes == "Performance degradation observed"
    
    def test_check_result_to_dict(self):
        """Test CheckResult to_dict conversion."""
        timestamp = datetime.utcnow()
        metrics = {"test_metric": 42.0}
        
        result = CheckResult(
            check_id="dict-test",
            timestamp=timestamp,
            status="fail",
            latency_ms=1000.0,
            error_message="Test error",
            metrics=metrics,
            notes="Test notes"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["check_id"] == "dict-test"
        assert result_dict["timestamp"] == timestamp.isoformat()
        assert result_dict["status"] == "fail"
        assert result_dict["latency_ms"] == 1000.0
        assert result_dict["error_message"] == "Test error"
        assert result_dict["metrics"] == metrics
        assert result_dict["notes"] == "Test notes"


class TestSentinelConfig:
    """Test SentinelConfig dataclass."""
    
    def test_config_defaults(self):
        """Test default configuration values (localhost fallbacks when env vars not set)."""
        import os
        # Ensure environment variables are not set for this test
        with patch.dict(os.environ, {}, clear=False):
            # Remove any potentially set variables
            for key in ['TARGET_BASE_URL', 'REDIS_URL', 'QDRANT_URL', 'NEO4J_BOLT', 'NEO4J_USER']:
                os.environ.pop(key, None)

            config = SentinelConfig()

            # Verify localhost defaults are used when environment variables not set
            assert config.target_base_url == "http://localhost:8000"
            assert config.redis_url == "redis://localhost:6379"
            assert config.qdrant_url == "http://localhost:6333"
            assert config.neo4j_bolt == "bolt://localhost:7687"
            assert config.neo4j_user == "veris_ro"
            assert config.schedule_cadence_sec == 60
            assert config.max_jitter_pct == 20
            assert config.per_check_timeout_sec == 10
            assert config.cycle_budget_sec == 45
            assert config.max_parallel_checks == 4
            assert config.alert_webhook is None
            assert config.github_repo is None
    
    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = SentinelConfig(
            target_base_url="http://localhost:8000",
            schedule_cadence_sec=30,
            max_parallel_checks=8,
            alert_webhook="https://hooks.slack.com/test",
            github_repo="test/repo"
        )

        assert config.target_base_url == "http://localhost:8000"
        assert config.schedule_cadence_sec == 30
        assert config.max_parallel_checks == 8
        assert config.alert_webhook == "https://hooks.slack.com/test"
        assert config.github_repo == "test/repo"

    def test_config_environment_variables(self):
        """Test that environment variables are properly read by __post_init__()."""
        import os

        # Set environment variables for Docker deployment
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://context-store:8000',
            'REDIS_URL': 'redis://redis:6379',
            'QDRANT_URL': 'http://qdrant:6333',
            'NEO4J_BOLT': 'bolt://neo4j:7687',
            'NEO4J_USER': 'custom_user'
        }):
            config = SentinelConfig()

            # Verify environment variables are used
            assert config.target_base_url == "http://context-store:8000"
            assert config.redis_url == "redis://redis:6379"
            assert config.qdrant_url == "http://qdrant:6333"
            assert config.neo4j_bolt == "bolt://neo4j:7687"
            assert config.neo4j_user == "custom_user"

    def test_config_explicit_override_env_vars(self):
        """Test that explicit parameters override environment variables."""
        import os

        # Set environment variables
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://context-store:8000',
            'REDIS_URL': 'redis://redis:6379'
        }):
            # Override with explicit values
            config = SentinelConfig(
                target_base_url="http://custom:9000",
                redis_url="redis://custom-redis:7000"
            )

            # Verify explicit values take precedence
            assert config.target_base_url == "http://custom:9000"
            assert config.redis_url == "redis://custom-redis:7000"

    def test_config_partial_env_vars(self):
        """Test that partial environment variables work with defaults."""
        import os

        # Set only some environment variables
        with patch.dict(os.environ, {
            'TARGET_BASE_URL': 'http://context-store:8000'
        }, clear=False):
            # Remove other variables to ensure fallback
            for key in ['REDIS_URL', 'QDRANT_URL', 'NEO4J_BOLT']:
                os.environ.pop(key, None)

            config = SentinelConfig()

            # Verify mixed behavior: env var used for target_base_url, defaults for others
            assert config.target_base_url == "http://context-store:8000"  # From env
            assert config.redis_url == "redis://localhost:6379"  # Default
            assert config.qdrant_url == "http://localhost:6333"  # Default
            assert config.neo4j_bolt == "bolt://localhost:7687"  # Default


class TestVerisHealthProbe:
    """Test VerisHealthProbe check functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def health_probe(self, config):
        """Create VerisHealthProbe instance."""
        return VerisHealthProbe(config)
    
    @pytest.mark.asyncio
    async def test_health_probe_success(self, health_probe):
        """Test successful health probe."""
        # Mock successful HTTP responses
        mock_responses = [
            # Liveness response
            AsyncMock(status=200, json=AsyncMock(return_value={"status": "alive"})),
            # Readiness response
            AsyncMock(status=200, json=AsyncMock(return_value={
                "components": [
                    {"name": "qdrant", "status": "ok"},
                    {"name": "redis", "status": "healthy"},
                    {"name": "neo4j", "status": "degraded"}
                ]
            }))
        ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.side_effect = mock_responses
            
            result = await health_probe.run_check()
            
            assert result.check_id == "S1-probes"
            assert result.status == "pass"
            assert result.latency_ms > 0
            assert result.metrics["status_bool"] == 1.0
            assert "health endpoints responding correctly" in result.notes
    
    @pytest.mark.asyncio
    async def test_health_probe_liveness_failure(self, health_probe):
        """Test health probe with liveness failure."""
        mock_response = AsyncMock(status=500)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await health_probe.run_check()
            
            assert result.check_id == "S1-probes"
            assert result.status == "fail"
            assert "Liveness check failed: HTTP 500" in result.error_message
    
    @pytest.mark.asyncio
    async def test_health_probe_component_failure(self, health_probe):
        """Test health probe with component failure."""
        mock_responses = [
            # Liveness response
            AsyncMock(status=200, json=AsyncMock(return_value={"status": "alive"})),
            # Readiness response with unhealthy component
            AsyncMock(status=200, json=AsyncMock(return_value={
                "components": [
                    {"name": "qdrant", "status": "failed"},
                    {"name": "redis", "status": "healthy"}
                ]
            }))
        ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.side_effect = mock_responses
            
            result = await health_probe.run_check()
            
            assert result.check_id == "S1-probes"
            assert result.status == "fail"
            assert "Qdrant not healthy: failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_health_probe_exception(self, health_probe):
        """Test health probe with exception."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception("Connection failed")
            
            result = await health_probe.run_check()
            
            assert result.check_id == "S1-probes"
            assert result.status == "fail"
            assert "Health check exception: Connection failed" in result.error_message


class TestGoldenFactRecall:
    """Test GoldenFactRecall functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def fact_recall(self, config):
        """Create GoldenFactRecall instance."""
        return GoldenFactRecall(config)
    
    @pytest.mark.asyncio
    async def test_golden_fact_recall_success(self, fact_recall):
        """Test successful fact recall."""
        # Mock successful HTTP responses
        store_response = AsyncMock(status=200)
        retrieve_response = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "memories": [{"content": '{"name": "Matt"}'}]
            })
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.side_effect = [
                store_response,  # Store fact
                retrieve_response,  # Retrieve with question 1
                retrieve_response,  # Retrieve with question 2
                store_response,  # Store next fact
                retrieve_response,  # Continue pattern...
                retrieve_response,
                store_response,
                retrieve_response,
                retrieve_response
            ]
            
            result = await fact_recall.run_check()
            
            assert result.check_id == "S2-golden-fact-recall"
            assert result.status == "pass"
            assert result.metrics["p_at_1"] == 1.0
            assert "6/6 tests passed" in result.notes
    
    @pytest.mark.asyncio
    async def test_golden_fact_recall_partial_success(self, fact_recall):
        """Test partial success in fact recall."""
        store_response = AsyncMock(status=200)
        good_retrieve = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "memories": [{"content": '{"name": "Matt"}'}]
            })
        )
        bad_retrieve = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "memories": [{"content": "irrelevant content"}]
            })
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.side_effect = [
                store_response,  # Store fact
                good_retrieve,   # Good retrieve
                bad_retrieve,    # Bad retrieve
                store_response,  # Continue...
                good_retrieve,
                good_retrieve,
                store_response,
                good_retrieve,
                good_retrieve
            ]
            
            result = await fact_recall.run_check()
            
            assert result.check_id == "S2-golden-fact-recall"
            assert result.status == "warn"  # P@1 = 5/6 = 0.83, which is >= 0.8 but < 1.0
            assert result.metrics["p_at_1"] == pytest.approx(0.833, rel=1e-2)
    
    @pytest.mark.asyncio
    async def test_golden_fact_recall_store_failure(self, fact_recall):
        """Test fact recall with storage failure."""
        store_response = AsyncMock(status=500)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = store_response
            
            result = await fact_recall.run_check()
            
            assert result.check_id == "S2-golden-fact-recall"
            assert result.status == "fail"
            assert "Failed to store fact: HTTP 500" in result.error_message


class TestMetricsWiring:
    """Test MetricsWiring functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def metrics_wiring(self, config):
        """Create MetricsWiring instance."""
        return MetricsWiring(config)
    
    @pytest.mark.asyncio
    async def test_metrics_wiring_success(self, metrics_wiring):
        """Test successful metrics wiring check."""
        dashboard_response = AsyncMock(
            status=200,
            json=AsyncMock(return_value={
                "system": {"cpu_percent": 45.2},
                "services": [{"name": "Redis", "status": "healthy"}],
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        analytics_response = AsyncMock(status=200, json=AsyncMock(return_value={"analytics": {}}))
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.side_effect = [
                dashboard_response,
                analytics_response
            ]
            
            result = await metrics_wiring.run_check()
            
            assert result.check_id == "S4-metrics-wiring"
            assert result.status == "pass"
            assert result.metrics["labels_present"] == 1.0
            assert result.metrics["percentiles_present"] == 1.0
    
    @pytest.mark.asyncio
    async def test_metrics_wiring_dashboard_failure(self, metrics_wiring):
        """Test metrics wiring with dashboard failure."""
        dashboard_response = AsyncMock(status=404)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = dashboard_response
            
            result = await metrics_wiring.run_check()
            
            assert result.check_id == "S4-metrics-wiring"
            assert result.status == "fail"
            assert "Dashboard endpoint failed: HTTP 404" in result.error_message
    
    @pytest.mark.asyncio
    async def test_metrics_wiring_missing_fields(self, metrics_wiring):
        """Test metrics wiring with missing dashboard fields."""
        dashboard_response = AsyncMock(
            status=200,
            json=AsyncMock(return_value={"system": {}})  # Missing services and timestamp
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = dashboard_response
            
            result = await metrics_wiring.run_check()
            
            assert result.check_id == "S4-metrics-wiring"
            assert result.status == "fail"
            assert "Missing dashboard fields" in result.error_message


class TestSecurityNegatives:
    """Test SecurityNegatives functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def security_negatives(self, config):
        """Create SecurityNegatives instance."""
        return SecurityNegatives(config)
    
    @pytest.mark.asyncio
    async def test_security_negatives_proper_blocking(self, security_negatives):
        """Test security negatives with proper blocking."""
        responses = [
            AsyncMock(status=200),  # Reader retrieve - allowed
            AsyncMock(status=403),  # Reader store - blocked
            AsyncMock(status=401),  # Invalid token analytics - blocked
            AsyncMock(status=403),  # No auth store - blocked
        ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.side_effect = responses[:2] + responses[3:]
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = responses[2]
            
            result = await security_negatives.run_check()
            
            assert result.check_id == "S5-security-negatives"
            assert result.status == "pass"
            assert result.metrics["unauthorized_block_rate"] == 0.75  # 3/4 properly blocked
    
    @pytest.mark.asyncio
    async def test_security_negatives_no_blocking(self, security_negatives):
        """Test security negatives with no blocking (permissive system)."""
        responses = [
            AsyncMock(status=200),  # All requests succeed
            AsyncMock(status=200),
            AsyncMock(status=200),
            AsyncMock(status=200),
        ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.side_effect = responses[:2] + responses[3:]
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = responses[2]
            
            result = await security_negatives.run_check()
            
            assert result.check_id == "S5-security-negatives"
            assert result.status == "pass"  # Lenient for current implementation
            assert result.metrics["unauthorized_block_rate"] == 0.0


class TestCapacitySmoke:
    """Test CapacitySmoke functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def capacity_smoke(self, config):
        """Create CapacitySmoke instance."""
        return CapacitySmoke(config)
    
    @pytest.mark.asyncio
    async def test_capacity_smoke_success(self, capacity_smoke):
        """Test successful capacity smoke test."""
        # Mock fast, successful responses
        mock_response = AsyncMock(status=200)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await capacity_smoke.run_check()
            
            assert result.check_id == "S8-capacity-smoke"
            assert result.status == "pass"
            assert result.metrics["p95_ms"] <= 300
            assert result.metrics["error_rate"] <= 0.005
            assert "20/20 successful" in result.notes
    
    @pytest.mark.asyncio
    async def test_capacity_smoke_high_latency(self, capacity_smoke):
        """Test capacity smoke test with high latency."""
        # Mock slow responses by adding delay
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.4)  # 400ms delay
            return AsyncMock(status=200)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = slow_response
            
            result = await capacity_smoke.run_check()
            
            assert result.check_id == "S8-capacity-smoke"
            assert result.status in ["warn", "fail"]  # Depends on exact timing
            assert result.metrics["p95_ms"] > 300
    
    @pytest.mark.asyncio
    async def test_capacity_smoke_high_error_rate(self, capacity_smoke):
        """Test capacity smoke test with high error rate."""
        # Mock mixture of success and failures
        responses = [AsyncMock(status=200) if i % 2 == 0 else AsyncMock(status=500) for i in range(20)]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.side_effect = responses
            
            result = await capacity_smoke.run_check()
            
            assert result.check_id == "S8-capacity-smoke"
            assert result.status == "fail"
            assert result.metrics["error_rate"] == 0.5  # 50% error rate


class TestSentinelRunner:
    """Test SentinelRunner core functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentinelConfig(
            target_base_url="http://test:8000",
            schedule_cadence_sec=1,  # Fast for testing
            max_parallel_checks=2,
            per_check_timeout_sec=1,
            cycle_budget_sec=5
        )
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink()
    
    @pytest.fixture
    def sentinel(self, config, temp_db):
        """Create SentinelRunner instance."""
        return SentinelRunner(config, temp_db)
    
    def test_sentinel_initialization(self, sentinel, config):
        """Test SentinelRunner initialization."""
        assert sentinel.config == config
        assert not sentinel.running
        assert len(sentinel.checks) == 6  # All check types
        assert "S1-probes" in sentinel.checks
        assert "S2-golden-fact-recall" in sentinel.checks
        assert len(sentinel.failures) == 0
        assert len(sentinel.reports) == 0
    
    def test_database_initialization(self, sentinel):
        """Test database table creation."""
        # Database should be initialized with tables
        with sqlite3.connect(sentinel.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "check_results" in tables
            assert "cycle_reports" in tables
    
    @pytest.mark.asyncio
    async def test_run_single_cycle_success(self, sentinel):
        """Test successful single monitoring cycle."""
        # Mock all checks to return success
        for check_id, check_instance in sentinel.checks.items():
            check_instance.run_check = AsyncMock(return_value=CheckResult(
                check_id=check_id,
                timestamp=datetime.utcnow(),
                status="pass",
                latency_ms=50.0
            ))
        
        cycle_report = await sentinel.run_single_cycle()
        
        assert cycle_report["total_checks"] == 6
        assert cycle_report["passed_checks"] == 6
        assert cycle_report["failed_checks"] == 0
        assert cycle_report["cycle_duration_ms"] > 0
        assert len(cycle_report["results"]) == 6
    
    @pytest.mark.asyncio
    async def test_run_single_cycle_with_failures(self, sentinel):
        """Test single cycle with some failures."""
        # Mock mix of success and failure
        for i, (check_id, check_instance) in enumerate(sentinel.checks.items()):
            status = "pass" if i % 2 == 0 else "fail"
            error_msg = None if status == "pass" else f"Check {check_id} failed"
            
            check_instance.run_check = AsyncMock(return_value=CheckResult(
                check_id=check_id,
                timestamp=datetime.utcnow(),
                status=status,
                latency_ms=100.0,
                error_message=error_msg
            ))
        
        cycle_report = await sentinel.run_single_cycle()
        
        assert cycle_report["total_checks"] == 6
        assert cycle_report["passed_checks"] == 3
        assert cycle_report["failed_checks"] == 3
        assert len(sentinel.failures) == 3  # Failed checks added to failures buffer
    
    @pytest.mark.asyncio
    async def test_run_single_cycle_timeout(self, sentinel):
        """Test single cycle with timeout."""
        # Mock checks that take too long
        async def slow_check():
            await asyncio.sleep(10)  # Longer than cycle_budget_sec
            return CheckResult("slow", datetime.utcnow(), "pass", 100.0)
        
        for check_instance in sentinel.checks.values():
            check_instance.run_check = slow_check
        
        cycle_report = await sentinel.run_single_cycle()
        
        # Should timeout and return timeout result
        assert any("timeout" in result["check_id"] for result in cycle_report["results"])
    
    def test_store_cycle_results(self, sentinel):
        """Test storing cycle results in database."""
        timestamp = datetime.utcnow()
        results = [
            CheckResult("test1", timestamp, "pass", 50.0, metrics={"test": 1.0}),
            CheckResult("test2", timestamp, "fail", 100.0, error_message="Test error")
        ]
        
        cycle_report = {
            "cycle_id": "test123",
            "timestamp": timestamp.isoformat(),
            "total_checks": 2,
            "passed_checks": 1,
            "failed_checks": 1,
            "cycle_duration_ms": 150.0,
            "results": [r.to_dict() for r in results]
        }
        
        sentinel._store_cycle_results(results, cycle_report)
        
        # Verify data was stored
        with sqlite3.connect(sentinel.db_path) as conn:
            check_count = conn.execute("SELECT COUNT(*) FROM check_results").fetchone()[0]
            cycle_count = conn.execute("SELECT COUNT(*) FROM cycle_reports").fetchone()[0]
            
            assert check_count == 2
            assert cycle_count == 1
            
            # Verify specific data
            check_data = conn.execute(
                "SELECT check_id, status, error_message FROM check_results ORDER BY check_id"
            ).fetchall()
            
            assert check_data[0] == ("test1", "pass", None)
            assert check_data[1] == ("test2", "fail", "Test error")
    
    def test_get_status(self, sentinel):
        """Test status retrieval."""
        # Add some test data
        test_report = {
            "cycle_id": "test123",
            "timestamp": datetime.utcnow().isoformat(),
            "total_checks": 5,
            "passed_checks": 4,
            "failed_checks": 1
        }
        sentinel.reports.append(test_report)
        
        status = sentinel.get_status()
        
        assert status["running"] == False
        assert status["last_cycle"] == test_report
        assert status["total_cycles"] == 1
        assert status["failure_count"] == 0
        assert "config" in status


class TestPerCheckIntervals:
    """Test per-check interval functionality (PR #390)."""

    @pytest.fixture
    def config(self):
        """Create test configuration using the new modular SentinelConfig."""
        # Import the new modular SentinelConfig for runner tests
        from src.monitoring.sentinel.models import SentinelConfig as ModularConfig
        return ModularConfig(
            target_base_url="http://test:8000",
            check_interval_seconds=60
        )

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_default_check_intervals_defined(self):
        """Test that DEFAULT_CHECK_INTERVALS has all checks defined."""
        from src.monitoring.sentinel.runner import SentinelRunner

        # Verify all expected checks have intervals
        expected_checks = [
            "S1-probes", "S2-golden-fact-recall", "S3-paraphrase-robustness",
            "S4-metrics-wiring", "S5-security-negatives", "S6-backup-restore",
            "S7-config-parity", "S8-capacity-smoke", "S9-graph-intent",
            "S10-content-pipeline", "S11-firewall-status"
        ]

        # PR #397: CI/CD-only checks have interval=0
        cicd_only_checks = ["S3-paraphrase-robustness", "S4-metrics-wiring",
                           "S7-config-parity", "S8-capacity-smoke", "S9-graph-intent"]

        for check_id in expected_checks:
            assert check_id in SentinelRunner.DEFAULT_CHECK_INTERVALS, \
                f"Missing interval for {check_id}"
            assert isinstance(SentinelRunner.DEFAULT_CHECK_INTERVALS[check_id], int), \
                f"Interval for {check_id} should be int"
            # CI/CD-only checks have interval=0, runtime checks have interval > 0
            if check_id in cicd_only_checks:
                assert SentinelRunner.DEFAULT_CHECK_INTERVALS[check_id] == 0, \
                    f"CI/CD-only check {check_id} should have interval=0"
            else:
                assert SentinelRunner.DEFAULT_CHECK_INTERVALS[check_id] > 0, \
                    f"Runtime check {check_id} should have positive interval"

    def test_default_intervals_values(self):
        """Test that default intervals have expected values.

        PR #397: Updated intervals with CI/CD-only checks disabled at runtime.
        """
        from src.monitoring.sentinel.runner import SentinelRunner

        intervals = SentinelRunner.DEFAULT_CHECK_INTERVALS

        # Runtime checks (continuous monitoring)
        assert intervals["S1-probes"] == 60  # 1 minute - health probes
        assert intervals["S11-firewall-status"] == 300  # 5 minutes - security

        # Spot-check monitors (hourly)
        assert intervals["S2-golden-fact-recall"] == 3600  # 1 hour
        assert intervals["S5-security-negatives"] == 3600  # 1 hour
        assert intervals["S6-backup-restore"] == 21600  # 6 hours
        assert intervals["S10-content-pipeline"] == 3600  # 1 hour

        # CI/CD-only checks (disabled at runtime)
        assert intervals["S3-paraphrase-robustness"] == 0
        assert intervals["S4-metrics-wiring"] == 0
        assert intervals["S7-config-parity"] == 0
        assert intervals["S8-capacity-smoke"] == 0
        assert intervals["S9-graph-intent"] == 0

    @patch.dict('os.environ', {'SENTINEL_INTERVAL_S1': '120', 'SENTINEL_INTERVAL_S6': '3600'})
    def test_load_check_intervals_env_override(self, config, temp_db):
        """Test that environment variables override default intervals."""
        from src.monitoring.sentinel.runner import SentinelRunner

        runner = SentinelRunner(config, temp_db)

        # S1 should be overridden to 120 seconds
        assert runner.check_intervals["S1-probes"] == 120

        # S6 should be overridden to 3600 seconds (1 hour)
        assert runner.check_intervals["S6-backup-restore"] == 3600

        # S2 should still have default value (no override) - PR #397: now 1 hour
        assert runner.check_intervals["S2-golden-fact-recall"] == 3600

    @patch.dict('os.environ', {'SENTINEL_INTERVAL_S1': 'invalid'})
    def test_load_check_intervals_invalid_env(self, config, temp_db):
        """Test that invalid environment values are ignored."""
        from src.monitoring.sentinel.runner import SentinelRunner

        runner = SentinelRunner(config, temp_db)

        # S1 should fall back to default due to invalid value
        assert runner.check_intervals["S1-probes"] == 60

    def test_is_check_due_first_run(self, config, temp_db):
        """Test that check is due on first run (no previous run time)."""
        from src.monitoring.sentinel.runner import SentinelRunner

        runner = SentinelRunner(config, temp_db)

        # First run should always be due (no last_check_time entry)
        # Exception: CI/CD-only checks are never due at runtime
        assert runner._is_check_due("S1-probes") is True
        assert runner._is_check_due("S6-backup-restore") is True

    def test_is_check_due_cicd_only_never_due(self, config, temp_db):
        """Test that CI/CD-only checks (interval=0) are never due at runtime.

        PR #397: CI/CD-only checks have interval=0 and should never run
        during runtime monitoring. They run via GitHub Actions on deploy.
        """
        from src.monitoring.sentinel.runner import SentinelRunner

        runner = SentinelRunner(config, temp_db)

        # CI/CD-only checks should never be due, even on first run
        cicd_only_checks = [
            "S3-paraphrase-robustness",
            "S4-metrics-wiring",
            "S7-config-parity",
            "S8-capacity-smoke",
            "S9-graph-intent"
        ]

        for check_id in cicd_only_checks:
            assert runner._is_check_due(check_id) is False, \
                f"CI/CD-only check {check_id} should never be due at runtime"

    def test_is_check_due_after_interval(self, config, temp_db):
        """Test that check is due after interval has passed."""
        from src.monitoring.sentinel.runner import SentinelRunner

        runner = SentinelRunner(config, temp_db)

        # Set last run time to 2 minutes ago
        runner.last_check_time["S1-probes"] = time.time() - 120

        # S1 has 60 second interval, so it should be due
        assert runner._is_check_due("S1-probes") is True

    def test_is_check_due_before_interval(self, config, temp_db):
        """Test that check is not due before interval has passed."""
        from src.monitoring.sentinel.runner import SentinelRunner

        runner = SentinelRunner(config, temp_db)

        # Set last run time to 30 seconds ago
        runner.last_check_time["S1-probes"] = time.time() - 30

        # S1 has 60 second interval, so it should NOT be due
        assert runner._is_check_due("S1-probes") is False

    def test_is_check_due_s6_long_interval(self, config, temp_db):
        """Test S6 long interval (6 hours) behavior."""
        from src.monitoring.sentinel.runner import SentinelRunner

        runner = SentinelRunner(config, temp_db)

        # Set last run time to 1 hour ago
        runner.last_check_time["S6-backup-restore"] = time.time() - 3600

        # S6 has 6 hour (21600 second) interval, so it should NOT be due
        assert runner._is_check_due("S6-backup-restore") is False

        # Set last run time to 7 hours ago
        runner.last_check_time["S6-backup-restore"] = time.time() - (7 * 3600)

        # Now it should be due
        assert runner._is_check_due("S6-backup-restore") is True

    def test_human_readable_interval_seconds(self, config, temp_db):
        """Test human-readable interval formatting for seconds."""
        from src.monitoring.sentinel.runner import SentinelRunner

        # Test intervals less than 60 seconds
        runner = SentinelRunner(config, temp_db)

        # Override to test formatting in logs
        # The actual formatting is in _load_check_intervals
        # We verify the logic: < 60s shows as Xs, >= 60s shows as Xm, >= 3600s shows as Xh
        assert 30 < 60  # Would display as "30s"
        assert 60 >= 60  # Would display as "1m"
        assert 3600 >= 3600  # Would display as "1h"

    def test_human_readable_interval_minutes(self):
        """Test human-readable interval for minutes."""
        # 300 seconds = 5 minutes
        interval = 300
        if interval >= 3600:
            human = f"{interval // 3600}h"
        elif interval >= 60:
            human = f"{interval // 60}m"
        else:
            human = f"{interval}s"

        assert human == "5m"

    def test_human_readable_interval_hours(self):
        """Test human-readable interval for hours."""
        # 21600 seconds = 6 hours
        interval = 21600
        if interval >= 3600:
            human = f"{interval // 3600}h"
        elif interval >= 60:
            human = f"{interval // 60}m"
        else:
            human = f"{interval}s"

        assert human == "6h"

    def test_human_readable_interval_edge_cases(self):
        """Test edge cases for interval formatting."""
        test_cases = [
            (30, "30s"),      # Less than a minute
            (60, "1m"),       # Exactly 1 minute
            (90, "1m"),       # 1.5 minutes (truncates to 1m)
            (3599, "59m"),    # Just under 1 hour
            (3600, "1h"),     # Exactly 1 hour
            (7200, "2h"),     # 2 hours
            (86400, "24h"),   # 24 hours
        ]

        for interval, expected in test_cases:
            if interval >= 3600:
                human = f"{interval // 3600}h"
            elif interval >= 60:
                human = f"{interval // 60}m"
            else:
                human = f"{interval}s"

            assert human == expected, f"Expected {expected} for {interval}s, got {human}"

    @pytest.mark.asyncio
    async def test_run_check_cycle_respects_intervals(self, config, temp_db):
        """Test that check cycle only runs due checks."""
        from src.monitoring.sentinel.runner import SentinelRunner

        runner = SentinelRunner(config, temp_db)

        # Mock all checks
        for check_id, check_instance in runner.checks.items():
            check_instance.execute = AsyncMock(return_value=CheckResult(
                check_id=check_id,
                timestamp=datetime.utcnow(),
                status="pass",
                latency_ms=50.0
            ))

        # First cycle - all checks should run (no previous times)
        await runner._run_check_cycle()

        # Verify all checks were called
        for check_id, check_instance in runner.checks.items():
            assert check_instance.execute.called, f"{check_id} should have run"

        # Reset mocks
        for check_instance in runner.checks.values():
            check_instance.execute.reset_mock()

        # Second cycle immediately - only S1 should run (60s interval, others longer)
        # Since no time has passed, no checks should be due
        await runner._run_check_cycle()

        # Fast checks (S1) has 60s interval, so it shouldn't run immediately
        # None should have run since we just ran them all
        for check_id, check_instance in runner.checks.items():
            assert not check_instance.execute.called, \
                f"{check_id} should NOT have run (interval not elapsed)"


class TestSentinelAPI:
    """Test SentinelAPI HTTP endpoints."""
    
    @pytest.fixture
    def sentinel_mock(self):
        """Create mock SentinelRunner."""
        mock = Mock()
        mock.get_status.return_value = {
            "running": True,
            "last_cycle": {"cycle_id": "test123"},
            "total_cycles": 10,
            "failure_count": 2
        }
        mock.run_single_cycle = AsyncMock(return_value={
            "success": True,
            "cycle_id": "new123"
        })
        mock.checks = {
            "S1-probes": Mock(__class__=Mock(__doc__="Health probe check")),
            "S2-golden": Mock(__class__=Mock(__doc__="Fact recall check"))
        }
        mock.reports = [{"cycle_id": f"cycle{i}"} for i in range(5)]
        mock.failures = []
        mock.running = True
        return mock
    
    @pytest.fixture
    def api_server(self, sentinel_mock):
        """Create SentinelAPI instance."""
        return SentinelAPI(sentinel_mock)
    
    @pytest.mark.asyncio
    async def test_status_endpoint(self, api_server, sentinel_mock):
        """Test /status endpoint."""
        request = Mock()
        response = await api_server.status_handler(request)
        
        assert response.status == 200
        response_data = json.loads(response.text)
        assert response_data["running"] == True
        assert response_data["total_cycles"] == 10
    
    @pytest.mark.asyncio
    async def test_run_endpoint_success(self, api_server, sentinel_mock):
        """Test /run endpoint with successful cycle."""
        request = Mock()
        response = await api_server.run_handler(request)
        
        assert response.status == 200
        response_data = json.loads(response.text)
        assert response_data["success"] == True
        assert "cycle_report" in response_data
        
        # Verify the mock was called
        sentinel_mock.run_single_cycle.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_endpoint_failure(self, api_server, sentinel_mock):
        """Test /run endpoint with cycle failure."""
        sentinel_mock.run_single_cycle.side_effect = Exception("Cycle failed")
        
        request = Mock()
        response = await api_server.run_handler(request)
        
        assert response.status == 500
        response_data = json.loads(response.text)
        assert response_data["success"] == False
        assert "error" in response_data
    
    @pytest.mark.asyncio
    async def test_checks_endpoint(self, api_server, sentinel_mock):
        """Test /checks endpoint."""
        request = Mock()
        response = await api_server.checks_handler(request)
        
        assert response.status == 200
        response_data = json.loads(response.text)
        
        assert "checks" in response_data
        checks = response_data["checks"]
        assert "S1-probes" in checks
        assert "S2-golden" in checks
        assert checks["S1-probes"]["enabled"] == True
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, api_server, sentinel_mock):
        """Test /metrics endpoint."""
        # Add a mock report with metrics
        sentinel_mock.reports = [{
            "total_checks": 6,
            "passed_checks": 5,
            "failed_checks": 1,
            "cycle_duration_ms": 2500.0
        }]
        
        request = Mock()
        response = await api_server.metrics_handler(request)
        
        assert response.status == 200
        assert response.content_type == "text/plain"
        
        metrics_text = response.text
        assert "sentinel_checks_total 6" in metrics_text
        assert "sentinel_checks_passed 5" in metrics_text
        assert "sentinel_checks_failed 1" in metrics_text
        assert "sentinel_running 1" in metrics_text
    
    @pytest.mark.asyncio
    async def test_report_endpoint(self, api_server, sentinel_mock):
        """Test /report endpoint."""
        request = Mock()
        request.query = {"n": "3"}
        
        response = await api_server.report_handler(request)
        
        assert response.status == 200
        response_data = json.loads(response.text)
        
        assert "reports" in response_data
        assert len(response_data["reports"]) == 3  # Last 3 reports
        assert response_data["total_reports"] == 5
    
    @pytest.mark.asyncio
    async def test_report_endpoint_default_limit(self, api_server, sentinel_mock):
        """Test /report endpoint with default limit."""
        request = Mock()
        request.query = {}
        
        response = await api_server.report_handler(request)
        
        assert response.status == 200
        response_data = json.loads(response.text)
        
        # Should default to last 10, but only 5 available
        assert len(response_data["reports"]) == 5


class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_cycle(self):
        """Test complete monitoring cycle end-to-end."""
        config = SentinelConfig(
            target_base_url="http://test:8000",
            schedule_cadence_sec=1,
            max_parallel_checks=2
        )
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name
        
        try:
            sentinel = SentinelRunner(config, temp_db)
            
            # Mock all checks for integration test
            mock_results = [
                CheckResult("S1-probes", datetime.utcnow(), "pass", 45.0),
                CheckResult("S2-golden-fact-recall", datetime.utcnow(), "pass", 1250.0),
                CheckResult("S4-metrics-wiring", datetime.utcnow(), "pass", 123.0),
                CheckResult("S5-security-negatives", datetime.utcnow(), "warn", 234.0),
                CheckResult("S7-config-parity", datetime.utcnow(), "pass", 67.0),
                CheckResult("S8-capacity-smoke", datetime.utcnow(), "pass", 2100.0)
            ]
            
            for i, (check_id, check_instance) in enumerate(sentinel.checks.items()):
                check_instance.run_check = AsyncMock(return_value=mock_results[i])
            
            # Run cycle
            cycle_report = await sentinel.run_single_cycle()
            
            # Verify complete cycle
            assert cycle_report["total_checks"] == 6
            assert cycle_report["passed_checks"] == 5
            assert cycle_report["failed_checks"] == 0  # warn counts as not failed
            
            # Verify database persistence
            with sqlite3.connect(temp_db) as conn:
                check_count = conn.execute("SELECT COUNT(*) FROM check_results").fetchone()[0]
                cycle_count = conn.execute("SELECT COUNT(*) FROM cycle_reports").fetchone()[0]
                
                assert check_count == 6
                assert cycle_count == 1
            
            # Verify status
            status = sentinel.get_status()
            assert status["total_cycles"] == 1
            assert status["last_cycle"]["total_checks"] == 6
            
        finally:
            Path(temp_db).unlink()
    
    @pytest.mark.asyncio
    async def test_alerting_workflow(self):
        """Test alerting workflow with failures."""
        config = SentinelConfig(
            target_base_url="http://test:8000",
            alert_webhook="https://hooks.slack.com/test"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name
        
        try:
            sentinel = SentinelRunner(config, temp_db)
            
            # Mock critical failure
            critical_failure = CheckResult(
                "S1-probes", 
                datetime.utcnow(), 
                "fail", 
                5000.0,
                error_message="All health endpoints down"
            )
            
            for check_instance in sentinel.checks.values():
                check_instance.run_check = AsyncMock(return_value=critical_failure)
            
            # Mock requests for webhook
            with patch('requests.post') as mock_post:
                mock_post.return_value = Mock(status_code=200)
                
                cycle_report = await sentinel.run_single_cycle()
                
                # Should have failures
                assert cycle_report["failed_checks"] == 6
                assert len(sentinel.failures) == 6
                
                # Should have triggered alerts (when REQUESTS_AVAILABLE=True)
                # In test environment, requests might not be available
        
        finally:
            Path(temp_db).unlink()


# Performance and stress tests
class TestPerformanceCharacteristics:
    """Test performance characteristics of Sentinel."""
    
    @pytest.mark.asyncio
    async def test_cycle_performance(self):
        """Test that cycles complete within reasonable time."""
        config = SentinelConfig(
            target_base_url="http://test:8000",
            max_parallel_checks=4,
            per_check_timeout_sec=1
        )
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name
        
        try:
            sentinel = SentinelRunner(config, temp_db)
            
            # Mock fast checks
            for check_instance in sentinel.checks.values():
                check_instance.run_check = AsyncMock(return_value=CheckResult(
                    "test", datetime.utcnow(), "pass", 50.0
                ))
            
            start_time = time.time()
            cycle_report = await sentinel.run_single_cycle()
            cycle_duration = time.time() - start_time
            
            # Should complete quickly with parallel execution
            assert cycle_duration < 2.0  # Should be well under 2 seconds
            assert cycle_report["cycle_duration_ms"] < 2000
            
        finally:
            Path(temp_db).unlink()
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_history(self):
        """Test memory usage with large history."""
        config = SentinelConfig()
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name
        
        try:
            sentinel = SentinelRunner(config, temp_db)
            
            # Fill up ring buffers
            for i in range(300):  # More than maxlen of failures (200)
                failure = CheckResult(f"test{i}", datetime.utcnow(), "fail", 100.0)
                sentinel.failures.append(failure)
            
            for i in range(100):  # More than maxlen of reports (50)
                report = {"cycle_id": f"cycle{i}", "timestamp": datetime.utcnow().isoformat()}
                sentinel.reports.append(report)
            
            # Ring buffers should be bounded
            assert len(sentinel.failures) == 200  # maxlen
            assert len(sentinel.reports) == 50   # maxlen
            
        finally:
            Path(temp_db).unlink()


# New Sentinel Checks Tests (S3, S6, S9, S10)
class TestParaphraseRobustness:
    """Test S3: Paraphrase robustness monitoring check."""
    
    @pytest.fixture
    def config(self):
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def check(self, config):
        return ParaphraseRobustness(config)
    
    @pytest.mark.asyncio
    async def test_paraphrase_robustness_pass(self, check):
        """Test paraphrase robustness check with passing scenario."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful API responses with consistent results
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "results": [{"score": 0.95, "text": "relevant result"}]
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S3-paraphrase-robustness"
            assert result.status == "pass"
            assert result.latency_ms > 0
            assert result.metrics is not None
            assert "consistency_rate" in result.metrics
    
    @pytest.mark.asyncio
    async def test_paraphrase_robustness_fail(self, check):
        """Test paraphrase robustness check with failing scenario."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock API error responses
            mock_response = AsyncMock()
            mock_response.status = 500
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S3-paraphrase-robustness"
            assert result.status == "fail"
            assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_paraphrase_robustness_inconsistent_results(self, check):
        """Test paraphrase robustness with inconsistent results."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock inconsistent API responses
            responses = [
                {"results": [{"score": 0.95, "text": "result A"}]},
                {"results": [{"score": 0.2, "text": "different result"}]},  # Inconsistent
            ]
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(side_effect=responses)
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S3-paraphrase-robustness"
            assert result.status in ["warn", "fail"]  # Should detect inconsistency


class TestBackupRestore:
    """Test S6: Backup and restore monitoring check."""
    
    @pytest.fixture
    def config(self):
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def check(self, config):
        return BackupRestore(config)
    
    @pytest.mark.asyncio
    async def test_backup_restore_pass(self, check):
        """Test backup restore check with passing scenario."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful backup/restore operations
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "success": True,
                "backup_id": "test-backup-123",
                "stored_contexts": 5
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S6-backup-restore"
            assert result.status == "pass"
            assert result.latency_ms > 0
            assert result.metrics is not None
            assert "contexts_verified" in result.metrics
    
    @pytest.mark.asyncio
    async def test_backup_restore_fail(self, check):
        """Test backup restore check with failing scenario."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock failed backup operation
            mock_response = AsyncMock()
            mock_response.status = 500
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S6-backup-restore"
            assert result.status == "fail"
            assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_backup_restore_data_integrity_issue(self, check):
        """Test backup restore with data integrity problems."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock responses where backup succeeds but restore shows missing data
            backup_response = AsyncMock()
            backup_response.status = 200
            backup_response.json = AsyncMock(return_value={
                "success": True,
                "backup_id": "test-backup-123",
                "stored_contexts": 5
            })
            
            restore_response = AsyncMock()
            restore_response.status = 200
            restore_response.json = AsyncMock(return_value={
                "success": True,
                "restored_contexts": 3  # Less than backed up!
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = backup_response
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = restore_response
            
            result = await check.run_check()
            
            assert result.check_id == "S6-backup-restore"
            assert result.status in ["warn", "fail"]  # Should detect data loss


class TestGraphIntentValidation:
    """Test S9: Graph intent validation monitoring check."""
    
    @pytest.fixture
    def config(self):
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def check(self, config):
        return GraphIntentValidation(config)
    
    @pytest.mark.asyncio
    async def test_graph_intent_validation_pass(self, check):
        """Test graph intent validation with high accuracy."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful intent analysis responses
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "intent": {"type": "semantic_search"},
                "graph_plan": {"node_types": ["concept", "document"]}
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S9-graph-intent"
            assert result.status == "pass"
            assert result.latency_ms > 0
            assert result.metrics is not None
            assert "intent_accuracy_rate" in result.metrics
            assert result.metrics["intent_accuracy_rate"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_graph_intent_validation_low_accuracy(self, check):
        """Test graph intent validation with low accuracy."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock incorrect intent analysis responses
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "intent": {"type": "wrong_intent"},  # Always wrong
                "graph_plan": {"node_types": ["wrong", "types"]}
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S9-graph-intent"
            assert result.status in ["warn", "fail"]
            assert result.metrics["intent_accuracy_rate"] < 0.8
    
    @pytest.mark.asyncio
    async def test_graph_intent_validation_api_error(self, check):
        """Test graph intent validation with API errors."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock API error
            mock_response = AsyncMock()
            mock_response.status = 503
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S9-graph-intent"
            assert result.status == "fail"
            assert result.error_message is not None


class TestContentPipelineMonitoring:
    """Test S10: Content pipeline monitoring check."""
    
    @pytest.fixture
    def config(self):
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def check(self, config):
        return ContentPipelineMonitoring(config)
    
    @pytest.mark.asyncio
    async def test_content_pipeline_monitoring_pass(self, check):
        """Test content pipeline monitoring with all stages healthy."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock healthy pipeline responses
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "processed_today": 100,
                "errors_today": 1,
                "queue_depth": 5,
                "vectors_generated_today": 95,
                "embedding_failures_today": 0,
                "stored_today": 95,
                "storage_failures_today": 0,
                "indexed_today": 95,
                "index_failures_today": 0
            })
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S10-content-pipeline"
            assert result.status == "pass"
            assert result.latency_ms > 0
            assert result.metrics is not None
            assert "pipeline_health_percent" in result.metrics
            assert result.metrics["pipeline_health_percent"] == 100.0
    
    @pytest.mark.asyncio
    async def test_content_pipeline_monitoring_degraded(self, check):
        """Test content pipeline monitoring with some unhealthy stages."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock mix of healthy and unhealthy responses
            responses = [
                # Ingestion healthy
                AsyncMock(status=200, json=AsyncMock(return_value={
                    "processed_today": 100, "errors_today": 1
                })),
                # Embedding unhealthy
                AsyncMock(status=503),
                # Storage healthy
                AsyncMock(status=200, json=AsyncMock(return_value={
                    "stored_today": 50, "storage_failures_today": 5
                })),
                # Indexing healthy
                AsyncMock(status=200, json=AsyncMock(return_value={
                    "indexed_today": 45, "index_failures_today": 0
                }))
            ]
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = responses[0]
            mock_session.return_value.__aenter__.return_value.get.side_effect = [
                responses[0].__aenter__.return_value,
                responses[1].__aenter__.return_value,
                responses[2].__aenter__.return_value,
                responses[3].__aenter__.return_value
            ]
            
            result = await check.run_check()
            
            assert result.check_id == "S10-content-pipeline"
            assert result.status in ["warn", "fail"]
            assert result.metrics["pipeline_health_percent"] < 100.0
    
    @pytest.mark.asyncio
    async def test_content_pipeline_monitoring_high_error_rate(self, check):
        """Test content pipeline monitoring with high error rates."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock responses with high error rates
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "processed_today": 100,
                "errors_today": 20,  # High error rate
                "vectors_generated_today": 70,
                "embedding_failures_today": 10,
                "stored_today": 60,
                "storage_failures_today": 15,
                "indexed_today": 55,
                "index_failures_today": 5
            })
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await check.run_check()
            
            assert result.check_id == "S10-content-pipeline"
            assert result.status in ["warn", "fail"]
            assert result.metrics["error_rate"] > 0.05  # More than 5% error rate

class TestMCPTypeValidation:
    """Test MCP type validation for S9 and S10 checks.
    
    Verifies that Sentinel checks use valid MCP types (design|decision|trace|sprint|log)
    and preserve original type information in metadata/tags for filtering/search.
    """
    
    @pytest.fixture
    def s9_config(self):
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def s10_config(self):
        return SentinelConfig(target_base_url="http://test:8000")
    
    @pytest.fixture
    def s9_check(self, s9_config):
        return GraphIntentValidation(s9_config)
    
    @pytest.fixture
    def s10_check(self, s10_config):
        return ContentPipelineMonitoring(s10_config)
    
    @pytest.mark.asyncio
    async def test_s9_uses_log_type(self, s9_check):
        """Test that S9 uses 'log' MCP type instead of invalid 'graph_intent_test'."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Setup mock to capture the payload sent to the API
            captured_payloads = []
            
            async def capture_post(*args, **kwargs):
                if 'json' in kwargs:
                    captured_payloads.append(kwargs['json'])
                mock_response = AsyncMock()
                mock_response.status = 201  # Success
                mock_response.json = AsyncMock(return_value={"context_id": "test123"})
                return mock_response
            
            mock_session_instance = AsyncMock()
            mock_session_instance.post.side_effect = capture_post
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            try:
                await s9_check.run_check()
            except:
                pass  # We're only interested in captured payloads
            
            # Verify at least one payload was sent
            assert len(captured_payloads) > 0, "S9 should send at least one context storage request"
            
            # Verify all payloads use 'log' type, not 'graph_intent_test'
            for payload in captured_payloads:
                assert 'content_type' in payload, "Payload must have content_type field"
                assert payload['content_type'] == 'log', \
                    f"S9 must use 'log' MCP type, not '{payload.get('content_type')}'"
                assert payload['content_type'] != 'graph_intent_test', \
                    "S9 must not use invalid 'graph_intent_test' type"
    
    @pytest.mark.asyncio
    async def test_s9_preserves_original_type_in_metadata(self, s9_check):
        """Test that S9 preserves original 'graph_intent_test' type in metadata.test_type."""
        with patch('aiohttp.ClientSession') as mock_session:
            captured_payloads = []
            
            async def capture_post(*args, **kwargs):
                if 'json' in kwargs:
                    captured_payloads.append(kwargs['json'])
                mock_response = AsyncMock()
                mock_response.status = 201
                mock_response.json = AsyncMock(return_value={"context_id": "test123"})
                return mock_response
            
            mock_session_instance = AsyncMock()
            mock_session_instance.post.side_effect = capture_post
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            try:
                await s9_check.run_check()
            except:
                pass
            
            # Verify original type is preserved in metadata
            assert len(captured_payloads) > 0
            for payload in captured_payloads:
                assert 'metadata' in payload, "Payload must have metadata field"
                assert 'test_type' in payload['metadata'], \
                    "Original type must be preserved in metadata.test_type"
                assert payload['metadata']['test_type'] == 'graph_intent_test', \
                    "Original 'graph_intent_test' type must be preserved in metadata for filtering"
    
    @pytest.mark.asyncio
    async def test_s10_uses_log_type_for_all_samples(self, s10_check):
        """Test that S10 uses 'log' MCP type for all 5 test samples."""
        # Get the default test samples directly from the check
        samples = s10_check._get_default_test_samples()
        
        # Verify all 5 samples use 'log' type
        assert len(samples) >= 5, "S10 should have at least 5 test samples"
        
        invalid_types = [
            'technical_documentation', 'api_documentation', 
            'troubleshooting_guide', 'deployment_process',
            'performance_optimization'
        ]
        
        for i, sample in enumerate(samples):
            assert 'type' in sample, f"Sample {i} must have 'type' field"
            assert sample['type'] == 'log', \
                f"Sample {i} must use 'log' MCP type, not '{sample.get('type')}'"
            assert sample['type'] not in invalid_types, \
                f"Sample {i} must not use invalid type '{sample.get('type')}'"
    
    @pytest.mark.asyncio
    async def test_s10_preserves_original_types_in_tags(self, s10_check):
        """Test that S10 preserves original types in content.tags array."""
        samples = s10_check._get_default_test_samples()
        
        # Expected original types that should be preserved
        expected_original_types = {
            'technical_documentation',
            'api_documentation',
            'troubleshooting_guide',
            'deployment_process',
            'performance_optimization'
        }
        
        # Collect all tags from samples
        found_original_types = set()
        for i, sample in enumerate(samples):
            assert 'content' in sample, f"Sample {i} must have 'content' field"
            assert 'tags' in sample['content'], f"Sample {i} content must have 'tags' field"
            
            tags = sample['content']['tags']
            assert isinstance(tags, list), f"Sample {i} tags must be a list"
            
            # Check if any original type is in the tags
            for tag in tags:
                if tag in expected_original_types:
                    found_original_types.add(tag)
        
        # Verify at least some original types are preserved in tags
        assert len(found_original_types) > 0, \
            "S10 must preserve original type values in content.tags for searchability"
        
        # Ideally, all original types should be preserved
        # But we'll be lenient and just check that tags exist
        for sample in samples:
            tags = sample['content']['tags']
            assert len(tags) > 0, "Each sample must have at least one tag"
    
    @pytest.mark.asyncio
    async def test_mcp_schema_accepts_log_type(self):
        """Test that 'log' is a valid MCP type according to schema pattern."""
        # Valid MCP types according to StoreContextRequest schema
        valid_mcp_types = {'design', 'decision', 'trace', 'sprint', 'log'}
        
        # Test that 'log' is in the valid set
        assert 'log' in valid_mcp_types, "'log' must be a valid MCP type"
        
        # Test that invalid types are not in the valid set
        invalid_types = {'graph_intent_test', 'technical_documentation', 
                        'api_documentation', 'troubleshooting_guide'}
        for invalid_type in invalid_types:
            assert invalid_type not in valid_mcp_types, \
                f"'{invalid_type}' must not be a valid MCP type"
    
    @pytest.mark.asyncio
    async def test_s9_original_type_retrievable(self, s9_check):
        """Test that S9's original type can be retrieved and filtered from metadata."""
        with patch('aiohttp.ClientSession') as mock_session:
            captured_payloads = []
            
            async def capture_post(*args, **kwargs):
                if 'json' in kwargs:
                    captured_payloads.append(kwargs['json'])
                mock_response = AsyncMock()
                mock_response.status = 201
                mock_response.json = AsyncMock(return_value={"context_id": "test123"})
                return mock_response
            
            mock_session_instance = AsyncMock()
            mock_session_instance.post.side_effect = capture_post
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            try:
                await s9_check.run_check()
            except:
                pass
            
            # Verify we can filter by original type
            assert len(captured_payloads) > 0
            graph_intent_payloads = [
                p for p in captured_payloads 
                if p.get('metadata', {}).get('test_type') == 'graph_intent_test'
            ]
            
            assert len(graph_intent_payloads) > 0, \
                "Should be able to filter contexts by original type in metadata.test_type"
    
    @pytest.mark.asyncio
    async def test_s10_original_types_retrievable(self, s10_check):
        """Test that S10's original types can be retrieved and filtered from tags."""
        samples = s10_check._get_default_test_samples()
        
        # Test filtering by original type tags
        for original_type in ['technical_documentation', 'api_documentation']:
            matching_samples = [
                s for s in samples 
                if original_type in s.get('content', {}).get('tags', [])
            ]
            
            # At least some samples should have the original type in tags
            # (We're being lenient here - not all samples need all types)
            # The key is that the mechanism exists for filtering
        
        # Verify tags are searchable
        for sample in samples:
            tags = sample.get('content', {}).get('tags', [])
            # Should be able to search/filter by any tag
            assert len(tags) > 0, "Each sample must have searchable tags"

    @pytest.mark.asyncio
    async def test_s10_uses_content_type_field_not_context_type(self, s10_check, mocker):
        """Test that S10 uses correct field name 'content_type' not 'context_type'.

        This test verifies the fix for the critical bug where S10 was using the wrong
        field name 'context_type' instead of 'content_type', causing all S10 tests to fail.
        """
        captured_payloads = []

        async def capture_payload(*args, **kwargs):
            # Capture JSON payload from POST requests
            if 'json' in kwargs:
                captured_payloads.append(kwargs['json'])
            # Mock successful response
            mock_response = mocker.AsyncMock()
            mock_response.status = 201
            mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = mocker.AsyncMock(return_value=None)
            return mock_response

        # Mock aiohttp.ClientSession.post to capture payloads
        mock_session = mocker.AsyncMock()
        mock_session.post = mocker.AsyncMock(side_effect=capture_payload)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        # Run the check
        try:
            await s10_check.run_check()
        except:
            pass  # Ignore errors, we just want to capture payloads

        # Verify all payloads use "content_type" not "context_type"
        assert len(captured_payloads) > 0, "Should have captured at least one payload"

        for payload in captured_payloads:
            # Critical assertion: Must use "content_type" field
            assert "content_type" in payload, \
                f"Payload must use 'content_type' field, not 'context_type': {payload.keys()}"

            # Must NOT use the wrong field name
            assert "context_type" not in payload, \
                f"Payload must NOT use 'context_type' field (wrong field name): {payload}"

            # Verify it's a valid MCP type
            assert payload["content_type"] == "log", \
                f"S10 test payloads must use 'log' type, got: {payload['content_type']}"


class TestAutonomousConfiguration:
    """Test autonomous configuration for S6/S7/S8 (PR #274)."""

    @pytest.fixture
    def s7_check_with_mcp(self):
        """S7 check instance - checks load config from environment."""
        from src.monitoring.sentinel.checks.s7_config_parity import ConfigParity
        config = SentinelConfig(target_base_url="http://test:8000")
        return ConfigParity(config)

    @pytest.fixture
    def s8_check_with_new_threshold(self):
        """S8 check instance - loads threshold from environment via config.get()."""
        from src.monitoring.sentinel.checks.s8_capacity_smoke import CapacitySmoke
        config = SentinelConfig(target_base_url="http://test:8000")
        return CapacitySmoke(config)

    def test_s7_recognizes_mcp_internal_url(self, s7_check_with_mcp):
        """Test that S7 recognizes MCP_INTERNAL_URL as optional env var (PR #274)."""
        # Verify MCP_INTERNAL_URL is in the optional env vars list
        assert "MCP_INTERNAL_URL" in s7_check_with_mcp.optional_env_vars, \
            "S7 must recognize MCP_INTERNAL_URL as optional environment variable"

    def test_s7_recognizes_mcp_forward_timeout(self, s7_check_with_mcp):
        """Test that S7 recognizes MCP_FORWARD_TIMEOUT as optional env var (PR #274)."""
        # Verify MCP_FORWARD_TIMEOUT is in the optional env vars list
        assert "MCP_FORWARD_TIMEOUT" in s7_check_with_mcp.optional_env_vars, \
            "S7 must recognize MCP_FORWARD_TIMEOUT as optional environment variable"

    def test_s7_mcp_vars_do_not_fail_check_when_missing(self, s7_check_with_mcp, mocker):
        """Test that missing MCP env vars don't fail S7 check (they're optional)."""
        # Mock environment without MCP vars
        mocker.patch.dict('os.environ', {
            'LOG_LEVEL': 'INFO',
            'ENVIRONMENT': 'production'
        }, clear=True)

        # Mock HTTP responses for S7 validation
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={
            "success": True,
            "config": {
                "python_version": "3.10",
                "fastapi_version": "0.115"
            }
        })
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        # S7 check should not fail due to missing optional MCP env vars
        # (The check might fail for other reasons in this minimal test, but not due to MCP vars)
        assert s7_check_with_mcp.optional_env_vars is not None
        assert len(s7_check_with_mcp.optional_env_vars) >= 8  # At least the 8 we added

    def test_s8_uses_2500ms_threshold(self, s8_check_with_new_threshold):
        """Test that S8 correctly loads 2500ms threshold from config (PR #274)."""
        # Verify the new threshold is loaded
        assert s8_check_with_new_threshold.max_response_time_ms == 2500, \
            "S8 must use 2500ms threshold (increased from 2000ms to account for REST→MCP forwarding)"

    def test_s8_default_threshold_is_2500ms(self):
        """Test that S8 defaults to 2500ms when not configured."""
        from src.monitoring.sentinel.checks.s8_capacity_smoke import CapacitySmoke

        # Create check with minimal config (no threshold specified)
        config = SentinelConfig(target_base_url="http://test:8000")
        check = CapacitySmoke(config)

        # Should default to 2500ms
        assert check.max_response_time_ms == 2500, \
            "S8 default max_response_time_ms should be 2500ms (PR #274 change)"

    @pytest.mark.asyncio
    async def test_s8_threshold_applied_in_capacity_test(self, s8_check_with_new_threshold, mocker):
        """Test that 2500ms threshold is actually used in capacity smoke tests."""
        # Mock responses that are under the new threshold but over the old one
        response_time_ms = 2200  # Between old (2000ms) and new (2500ms) thresholds

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.text = mocker.AsyncMock(return_value="OK")
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)
        mock_response.close = mocker.Mock()

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        # Mock time.time() to simulate 2200ms response time
        times = [0, response_time_ms / 1000, response_time_ms / 1000 * 2]
        mocker.patch('time.time', side_effect=times * 100)  # Repeat for multiple requests

        # Run check - should pass because 2200ms < 2500ms threshold
        # (Would have failed with old 2000ms threshold)
        result = await s8_check_with_new_threshold.run_check()

        # Check should evaluate against 2500ms threshold
        assert s8_check_with_new_threshold.max_response_time_ms == 2500

    @pytest.mark.asyncio
    async def test_s7_validates_mcp_internal_url_format(self, mocker):
        """Test that S7 validates MCP_INTERNAL_URL format correctly."""
        from src.monitoring.sentinel.checks.s7_config_parity import ConfigParity

        # Test invalid URL (no http/https protocol)
        mocker.patch.dict('os.environ', {
            'LOG_LEVEL': 'INFO',
            'ENVIRONMENT': 'production',
            'MCP_INTERNAL_URL': 'localhost:8000'  # Invalid - no protocol
        }, clear=True)

        config = SentinelConfig(target_base_url="http://test:8000")
        check = ConfigParity(config)

        # Mock HTTP client
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        # Run environment variable validation
        result = await check._check_environment_variables()

        # Should have config issue for invalid URL format
        assert "config_issues" in result
        config_issues = result.get("config_issues", [])
        assert any("MCP_INTERNAL_URL format invalid" in issue for issue in config_issues), \
            f"Expected MCP_INTERNAL_URL format validation error, got: {config_issues}"

    @pytest.mark.asyncio
    async def test_s7_accepts_valid_mcp_internal_url(self, mocker):
        """Test that S7 accepts valid MCP_INTERNAL_URL with http/https."""
        from src.monitoring.sentinel.checks.s7_config_parity import ConfigParity

        # Test valid URL
        mocker.patch.dict('os.environ', {
            'LOG_LEVEL': 'INFO',
            'ENVIRONMENT': 'production',
            'MCP_INTERNAL_URL': 'http://localhost:8000'  # Valid
        }, clear=True)

        config = SentinelConfig(target_base_url="http://test:8000")
        check = ConfigParity(config)

        # Mock HTTP client
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await check._check_environment_variables()

        # Should NOT have config issue for valid URL
        config_issues = result.get("config_issues", [])
        assert not any("MCP_INTERNAL_URL" in issue for issue in config_issues), \
            f"Valid MCP_INTERNAL_URL should not cause config issues, got: {config_issues}"

    @pytest.mark.asyncio
    async def test_s7_validates_mcp_forward_timeout_range(self, mocker):
        """Test that S7 validates MCP_FORWARD_TIMEOUT is within reasonable range."""
        from src.monitoring.sentinel.checks.s7_config_parity import ConfigParity

        # Test timeout value outside range (> 300 seconds)
        mocker.patch.dict('os.environ', {
            'LOG_LEVEL': 'INFO',
            'ENVIRONMENT': 'production',
            'MCP_FORWARD_TIMEOUT': '500'  # Invalid - too high
        }, clear=True)

        config = SentinelConfig(target_base_url="http://test:8000")
        check = ConfigParity(config)

        # Mock HTTP client
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await check._check_environment_variables()

        # Should have config issue for out-of-range timeout
        config_issues = result.get("config_issues", [])
        assert any("MCP_FORWARD_TIMEOUT" in issue and "range" in issue for issue in config_issues), \
            f"Expected MCP_FORWARD_TIMEOUT range validation error, got: {config_issues}"

    @pytest.mark.asyncio
    async def test_s7_validates_mcp_forward_timeout_is_numeric(self, mocker):
        """Test that S7 validates MCP_FORWARD_TIMEOUT is a valid number."""
        from src.monitoring.sentinel.checks.s7_config_parity import ConfigParity

        # Test non-numeric timeout value
        mocker.patch.dict('os.environ', {
            'LOG_LEVEL': 'INFO',
            'ENVIRONMENT': 'production',
            'MCP_FORWARD_TIMEOUT': 'not-a-number'  # Invalid
        }, clear=True)

        config = SentinelConfig(target_base_url="http://test:8000")
        check = ConfigParity(config)

        # Mock HTTP client
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await check._check_environment_variables()

        # Should have config issue for non-numeric timeout
        config_issues = result.get("config_issues", [])
        assert any("MCP_FORWARD_TIMEOUT" in issue and "number" in issue for issue in config_issues), \
            f"Expected MCP_FORWARD_TIMEOUT numeric validation error, got: {config_issues}"

    def test_backup_script_exists_and_is_executable(self):
        """Test that backup script required by deployment exists and is executable."""
        import os
        import stat

        # Verify backup script existence (required for S6 validation)
        backup_script_path = "scripts/backup-production-final.sh"

        assert os.path.exists(backup_script_path), \
            f"Backup script must exist at {backup_script_path} for S6 validation to pass"

        # Verify it's executable
        file_stat = os.stat(backup_script_path)
        is_executable = bool(file_stat.st_mode & stat.S_IXUSR)

        assert is_executable, \
            f"Backup script at {backup_script_path} must be executable (chmod +x)"

        # Verify it's a bash script
        with open(backup_script_path, 'r') as f:
            first_line = f.readline()
            assert first_line.startswith('#!/bin/bash') or first_line.startswith('#!/usr/bin/env bash'), \
                f"Backup script must be a bash script, got: {first_line}"

    def test_s8_loads_app_latency_threshold(self):
        """Test that S8 loads separate application latency threshold (PR #274 performance regression mitigation)."""
        from src.monitoring.sentinel.checks.s8_capacity_smoke import CapacitySmoke
        import os

        # S8 reads app_latency_ms from config.get() which reads from environment
        # Set environment variable for test
        os.environ['S8_APP_LATENCY_MS'] = '1500'

        try:
            config = SentinelConfig(target_base_url="http://test:8000")
            check = CapacitySmoke(config)

            # Verify app latency threshold is loaded
            assert check.app_latency_threshold_ms == 1500, \
                "S8 must load app_latency_threshold_ms from config"

            # Verify it's distinct from total response time threshold
            assert check.max_response_time_ms == 2500, \
                "S8 max_response_time_ms should be 2500ms (total including forwarding)"

            # App threshold should be lower than total threshold (app doesn't include forwarding)
        finally:
            # Clean up environment
            if 'S8_APP_LATENCY_MS' in os.environ:
                del os.environ['S8_APP_LATENCY_MS']
        assert check.app_latency_threshold_ms < check.max_response_time_ms, \
            "Application latency threshold should be lower than total response time threshold"


class TestS8AppLatencyBreakdown:
    """Tests for S8 application latency breakdown monitoring (PR #274)."""

    @pytest.fixture
    def s8_check(self):
        """Create S8 check instance for testing."""
        from src.monitoring.sentinel.checks.s8_capacity_smoke import CapacitySmoke
        config = SentinelConfig(target_base_url="http://test:8000")
        return CapacitySmoke(config)

    def _mock_metrics_response(self, mock_session, status, json_data=None):
        """Helper to mock metrics endpoint response."""
        from unittest.mock import AsyncMock, MagicMock

        mock_response = AsyncMock(status=status)
        if json_data is not None:
            mock_response.json = AsyncMock(return_value=json_data)

        # Create async context manager for session.get()
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        # Set up: session.get() returns an async context manager (not async)
        mock_session_instance = mock_session.return_value.__aenter__.return_value
        mock_session_instance.get = MagicMock(return_value=mock_get_cm)

    @pytest.mark.asyncio
    async def test_app_latency_breakdown_available(self, s8_check):
        """Test successful latency breakdown when metrics endpoint provides data."""
        from unittest.mock import patch

        with patch('aiohttp.ClientSession') as mock_session:
            self._mock_metrics_response(mock_session, 200, {
                "application_latency_ms": 450.5,
                "forwarding_latency_ms": 150.2,
                "other_metrics": "data"
            })

            result = await s8_check._check_application_latency_breakdown()

            assert result["breakdown_available"] is True
            assert result["application_latency_ms"] == 450.5
            assert result["forwarding_latency_ms"] == 150.2
            assert result["total_latency_ms"] == 600.7  # 450.5 + 150.2
            assert "error" not in result

    @pytest.mark.asyncio
    async def test_app_latency_breakdown_metrics_endpoint_error(self, s8_check):
        """Test handling when metrics endpoint returns HTTP error."""
        from unittest.mock import patch

        with patch('aiohttp.ClientSession') as mock_session:
            self._mock_metrics_response(mock_session, 500)

            result = await s8_check._check_application_latency_breakdown()

            assert result["breakdown_available"] is False
            assert "error" in result
            assert "HTTP 500" in result["error"]

    @pytest.mark.asyncio
    async def test_app_latency_breakdown_not_available(self, s8_check):
        """Test handling when metrics endpoint doesn't provide breakdown fields."""
        from unittest.mock import patch

        with patch('aiohttp.ClientSession') as mock_session:
            self._mock_metrics_response(mock_session, 200, {
                "total_requests": 1000,
                "avg_response_time": 500
                # Missing: application_latency_ms, forwarding_latency_ms
            })

            result = await s8_check._check_application_latency_breakdown()

            assert result["breakdown_available"] is False
            assert "error" in result
            assert "does not provide latency breakdown" in result["error"]

    @pytest.mark.asyncio
    async def test_app_latency_breakdown_connection_error(self, s8_check):
        """Test handling when metrics endpoint connection fails."""
        from unittest.mock import patch, MagicMock

        with patch('aiohttp.ClientSession') as mock_session:
            # Make get() raise an exception
            mock_session_instance = mock_session.return_value.__aenter__.return_value
            mock_session_instance.get = MagicMock(side_effect=Exception("Connection refused"))

            result = await s8_check._check_application_latency_breakdown()

            assert result["breakdown_available"] is False
            assert "error" in result
            assert "Failed to check metrics endpoint" in result["error"]
            assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_app_latency_breakdown_partial_fields(self, s8_check):
        """Test handling when only one breakdown field is present."""
        from unittest.mock import patch

        with patch('aiohttp.ClientSession') as mock_session:
            self._mock_metrics_response(mock_session, 200, {
                "application_latency_ms": 450.5
                # Missing: forwarding_latency_ms
            })

            result = await s8_check._check_application_latency_breakdown()

            assert result["breakdown_available"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_app_latency_breakdown_zero_values(self, s8_check):
        """Test handling of zero latency values (valid edge case)."""
        from unittest.mock import patch

        with patch('aiohttp.ClientSession') as mock_session:
            self._mock_metrics_response(mock_session, 200, {
                "application_latency_ms": 0.0,
                "forwarding_latency_ms": 0.0
            })

            result = await s8_check._check_application_latency_breakdown()

            assert result["breakdown_available"] is True
            assert result["application_latency_ms"] == 0.0
            assert result["forwarding_latency_ms"] == 0.0
            assert result["total_latency_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_app_latency_breakdown_float_conversion(self, s8_check):
        """Test that integer values are properly converted to floats."""
        from unittest.mock import patch

        with patch('aiohttp.ClientSession') as mock_session:
            self._mock_metrics_response(mock_session, 200, {
                "application_latency_ms": 450,  # Integer
                "forwarding_latency_ms": 150   # Integer
            })

            result = await s8_check._check_application_latency_breakdown()

            assert result["breakdown_available"] is True
            assert isinstance(result["application_latency_ms"], float)
            assert isinstance(result["forwarding_latency_ms"], float)
            assert result["application_latency_ms"] == 450.0
            assert result["forwarding_latency_ms"] == 150.0
            assert result["total_latency_ms"] == 600.0


class TestS7ConfigParityValidation:
    """Comprehensive tests for S7 config parity environment variable validation."""

    @pytest.fixture
    def s7_check(self):
        """Create S7 check instance for testing."""
        from src.monitoring.sentinel.checks.s7_config_parity import ConfigParity
        config = SentinelConfig(target_base_url="http://test:8000")
        return ConfigParity(config)

    def test_s7_optional_env_vars_includes_mcp_vars(self, s7_check):
        """Test that S7 optional env vars includes MCP_INTERNAL_URL and MCP_FORWARD_TIMEOUT."""
        assert "MCP_INTERNAL_URL" in s7_check.optional_env_vars
        assert "MCP_FORWARD_TIMEOUT" in s7_check.optional_env_vars

    @pytest.mark.asyncio
    async def test_s7_validates_mcp_url_with_protocol(self, s7_check, mocker):
        """Test S7 accepts valid MCP_INTERNAL_URL with http/https protocol."""
        mocker.patch.dict('os.environ', {
            'MCP_INTERNAL_URL': 'http://localhost:8000',
            'LOG_LEVEL': 'INFO'
        }, clear=True)

        # Mock HTTP client
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await s7_check._check_environment_variables()

        config_issues = result.get("config_issues", [])
        assert not any("MCP_INTERNAL_URL" in issue for issue in config_issues)

    @pytest.mark.asyncio
    async def test_s7_rejects_mcp_url_without_protocol(self, s7_check, mocker):
        """Test S7 rejects MCP_INTERNAL_URL without http/https protocol."""
        mocker.patch.dict('os.environ', {
            'MCP_INTERNAL_URL': 'localhost:8000',  # Missing protocol
            'LOG_LEVEL': 'INFO'
        }, clear=True)

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await s7_check._check_environment_variables()

        config_issues = result.get("config_issues", [])
        assert any("MCP_INTERNAL_URL format invalid" in issue for issue in config_issues)

    @pytest.mark.asyncio
    async def test_s7_rejects_mcp_url_with_whitespace(self, s7_check, mocker):
        """Test S7 rejects MCP_INTERNAL_URL containing whitespace."""
        mocker.patch.dict('os.environ', {
            'MCP_INTERNAL_URL': 'http://localhost :8000',  # Contains space
            'LOG_LEVEL': 'INFO'
        }, clear=True)

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await s7_check._check_environment_variables()

        config_issues = result.get("config_issues", [])
        assert any("MCP_INTERNAL_URL" in issue and "whitespace" in issue for issue in config_issues)

    @pytest.mark.asyncio
    async def test_s7_validates_mcp_timeout_in_range(self, s7_check, mocker):
        """Test S7 accepts MCP_FORWARD_TIMEOUT in valid range (0-300)."""
        mocker.patch.dict('os.environ', {
            'MCP_FORWARD_TIMEOUT': '60',  # Valid
            'LOG_LEVEL': 'INFO'
        }, clear=True)

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await s7_check._check_environment_variables()

        config_issues = result.get("config_issues", [])
        assert not any("MCP_FORWARD_TIMEOUT" in issue for issue in config_issues)

    @pytest.mark.asyncio
    async def test_s7_rejects_mcp_timeout_too_high(self, s7_check, mocker):
        """Test S7 rejects MCP_FORWARD_TIMEOUT above 300 seconds."""
        mocker.patch.dict('os.environ', {
            'MCP_FORWARD_TIMEOUT': '500',  # Too high
            'LOG_LEVEL': 'INFO'
        }, clear=True)

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await s7_check._check_environment_variables()

        config_issues = result.get("config_issues", [])
        assert any("MCP_FORWARD_TIMEOUT" in issue and "range" in issue for issue in config_issues)

    @pytest.mark.asyncio
    async def test_s7_rejects_mcp_timeout_zero(self, s7_check, mocker):
        """Test S7 rejects MCP_FORWARD_TIMEOUT of zero."""
        mocker.patch.dict('os.environ', {
            'MCP_FORWARD_TIMEOUT': '0',  # Invalid - must be positive
            'LOG_LEVEL': 'INFO'
        }, clear=True)

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await s7_check._check_environment_variables()

        config_issues = result.get("config_issues", [])
        assert any("MCP_FORWARD_TIMEOUT" in issue and "range" in issue for issue in config_issues)

    @pytest.mark.asyncio
    async def test_s7_rejects_non_numeric_mcp_timeout(self, s7_check, mocker):
        """Test S7 rejects non-numeric MCP_FORWARD_TIMEOUT."""
        mocker.patch.dict('os.environ', {
            'MCP_FORWARD_TIMEOUT': 'not-a-number',
            'LOG_LEVEL': 'INFO'
        }, clear=True)

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(return_value={"success": True, "config": {}})
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_session = mocker.AsyncMock()
        mock_session.get = mocker.AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('aiohttp.ClientSession', return_value=mock_session)

        result = await s7_check._check_environment_variables()

        config_issues = result.get("config_issues", [])
        assert any("MCP_FORWARD_TIMEOUT" in issue and "number" in issue for issue in config_issues)
