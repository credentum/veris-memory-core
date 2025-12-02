"""
Comprehensive monitoring and rate limiter tests for maximum coverage improvement.

Tests TokenBucket, RateLimiter, MetricsCollector, and AlertManager with
comprehensive coverage of all methods, error conditions, and edge cases.
"""

import tempfile
import time
from datetime import datetime, timedelta

import pytest
import yaml

from src.core.rate_limiter import TokenBucket


class TestTokenBucket:
    """Comprehensive tests for TokenBucket algorithm."""

    def test_token_bucket_initialization(self):
        """Test TokenBucket initialization with valid parameters."""
        bucket = TokenBucket(capacity=10, refill_rate=5)

        assert bucket.capacity == 10
        assert bucket.refill_rate == 5
        assert bucket.tokens == 10  # Should start full
        assert bucket.last_update is not None

    def test_token_bucket_initialization_zero_capacity(self):
        """Test TokenBucket initialization with zero capacity."""
        bucket = TokenBucket(capacity=0, refill_rate=5)

        assert bucket.capacity == 0
        assert bucket.tokens == 0

    def test_token_bucket_initialization_zero_refill_rate(self):
        """Test TokenBucket initialization with zero refill rate."""
        bucket = TokenBucket(capacity=10, refill_rate=0)

        assert bucket.refill_rate == 0

    def test_token_bucket_consume_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=5)

        result = bucket.consume(5)

        assert result is True
        assert bucket.tokens == 5

    def test_token_bucket_consume_insufficient_tokens(self):
        """Test token consumption when insufficient tokens available."""
        bucket = TokenBucket(capacity=10, refill_rate=5)
        bucket.tokens = 3  # Set to fewer tokens

        result = bucket.consume(5)

        assert result is False
        assert bucket.tokens == 3  # Unchanged

    def test_token_bucket_consume_zero_tokens(self):
        """Test consuming zero tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=5)

        result = bucket.consume(0)

        assert result is True
        assert bucket.tokens == 10  # Unchanged

    def test_token_bucket_consume_negative_tokens(self):
        """Test consuming negative tokens raises error."""
        bucket = TokenBucket(capacity=10, refill_rate=5)

        with pytest.raises(ValueError):
            bucket.consume(-1)

    def test_token_bucket_refill(self):
        """Test token refill mechanism."""
        bucket = TokenBucket(capacity=10, refill_rate=5)
        bucket.tokens = 0
        bucket.last_refill = time.time() - 2  # 2 seconds ago

        bucket._refill()

        # Should refill 5 tokens/second * 2 seconds = 10 tokens
        assert bucket.tokens == 10

    def test_token_bucket_refill_partial(self):
        """Test partial token refill."""
        bucket = TokenBucket(capacity=10, refill_rate=5)
        bucket.tokens = 5
        bucket.last_refill = time.time() - 0.6  # 0.6 seconds ago

        bucket._refill()

        # Should refill 5 tokens/second * 0.6 seconds = 3 tokens
        # 5 + 3 = 8 tokens
        assert bucket.tokens == 8

    def test_token_bucket_refill_exceeds_capacity(self):
        """Test refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=5)
        bucket.tokens = 8
        bucket.last_refill = time.time() - 2  # Would refill 10 tokens

        bucket._refill()

        # Should cap at capacity
        assert bucket.tokens == 10

    def test_token_bucket_available_tokens(self):
        """Test getting available tokens count."""
        bucket = TokenBucket(capacity=10, refill_rate=5)
        bucket.tokens = 7

        available = bucket.available_tokens()

        assert available == 7

    def test_token_bucket_available_tokens_with_refill(self):
        """Test available tokens includes refilled tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=5)
        bucket.tokens = 2
        bucket.last_refill = time.time() - 1  # 1 second ago, should refill 5

        available = bucket.available_tokens()

        assert available == 7  # 2 + 5


class TestRateLimiter:
    """Comprehensive tests for RateLimiter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_config_dir, "test_config.yaml")

        # Create test config
        test_config = {
            "rate_limiting": {"requests_per_minute": 60, "burst_size": 10, "enabled": True}
        }

        with open(self.config_path, "w") as f:
            yaml.dump(test_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_config_dir, ignore_errors=True)

    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(self.config_path)

        assert limiter.enabled is True
        assert limiter.buckets == {}

    def test_rate_limiter_initialization_disabled(self):
        """Test RateLimiter initialization when disabled."""
        config = {"rate_limiting": {"enabled": False}}

        config_path = os.path.join(self.test_config_dir, "disabled_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        limiter = RateLimiter(config_path)

        assert limiter.enabled is False

    def test_rate_limiter_check_limit_success(self):
        """Test successful rate limit check."""
        limiter = RateLimiter(self.config_path)

        result = limiter.check_limit("test_key")

        assert result is True

    def test_rate_limiter_check_limit_disabled(self):
        """Test rate limit check when disabled."""
        config = {"rate_limiting": {"enabled": False}}

        config_path = os.path.join(self.test_config_dir, "disabled_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        limiter = RateLimiter(config_path)

        result = limiter.check_limit("test_key")

        assert result is True  # Always true when disabled

    def test_rate_limiter_check_limit_creates_bucket(self):
        """Test that check_limit creates bucket for new key."""
        limiter = RateLimiter(self.config_path)

        assert "test_key" not in limiter.buckets

        limiter.check_limit("test_key")

        assert "test_key" in limiter.buckets

    def test_rate_limiter_check_limit_exhausted(self):
        """Test rate limit check when bucket is exhausted."""
        limiter = RateLimiter(self.config_path)

        # Create bucket with no tokens
        bucket = TokenBucket(capacity=10, refill_rate=1)
        bucket.tokens = 0
        limiter.buckets["test_key"] = bucket

        result = limiter.check_limit("test_key")

        assert result is False

    def test_rate_limiter_reset_limit(self):
        """Test resetting rate limit for a key."""
        limiter = RateLimiter(self.config_path)

        # Create bucket and consume tokens
        limiter.check_limit("test_key")
        limiter.buckets["test_key"].tokens = 0

        limiter.reset_limit("test_key")

        assert limiter.buckets["test_key"].tokens == 10  # Reset to capacity

    def test_rate_limiter_reset_limit_nonexistent_key(self):
        """Test resetting rate limit for non-existent key."""
        limiter = RateLimiter(self.config_path)

        # Should not raise exception
        limiter.reset_limit("nonexistent_key")

    def test_rate_limiter_get_remaining_tokens(self):
        """Test getting remaining tokens for a key."""
        limiter = RateLimiter(self.config_path)

        limiter.check_limit("test_key")
        limiter.buckets["test_key"].tokens = 7

        remaining = limiter.get_remaining_tokens("test_key")

        assert remaining == 7

    def test_rate_limiter_get_remaining_tokens_nonexistent_key(self):
        """Test getting remaining tokens for non-existent key."""
        limiter = RateLimiter(self.config_path)

        remaining = limiter.get_remaining_tokens("nonexistent_key")

        assert remaining == 0

    def test_rate_limiter_cleanup_expired_buckets(self):
        """Test cleanup of expired buckets."""
        limiter = RateLimiter(self.config_path)
        limiter.cleanup_interval = 1  # 1 second for testing

        # Create old bucket
        old_bucket = TokenBucket(capacity=10, refill_rate=1)
        old_bucket.last_refill = time.time() - 3600  # 1 hour ago
        limiter.buckets["old_key"] = old_bucket

        # Create recent bucket
        recent_bucket = TokenBucket(capacity=10, refill_rate=1)
        limiter.buckets["recent_key"] = recent_bucket

        limiter._cleanup_expired_buckets()

        assert "old_key" not in limiter.buckets
        assert "recent_key" in limiter.buckets


class TestMetricsCollector:
    """Comprehensive tests for MetricsCollector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_config_dir, "test_config.yaml")

        # Create test config
        test_config = {
            "monitoring": {"enabled": True, "metrics_retention_days": 7, "collection_interval": 60}
        }

        with open(self.config_path, "w") as f:
            yaml.dump(test_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_config_dir, ignore_errors=True)

    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector(self.config_path)

        assert collector.enabled is True
        assert collector.metrics == {}

    def test_metrics_collector_initialization_disabled(self):
        """Test MetricsCollector initialization when disabled."""
        config = {"monitoring": {"enabled": False}}

        config_path = os.path.join(self.test_config_dir, "disabled_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        collector = MetricsCollector(config_path)

        assert collector.enabled is False

    def test_metrics_collector_record_metric(self):
        """Test recording a metric."""
        collector = MetricsCollector(self.config_path)

        collector.record_metric("test_metric", 42.5, {"tag": "value"})

        assert "test_metric" in collector.metrics
        metric_data = collector.metrics["test_metric"][-1]
        assert metric_data["value"] == 42.5
        assert metric_data["tags"] == {"tag": "value"}

    def test_metrics_collector_record_metric_disabled(self):
        """Test recording metric when disabled."""
        config = {"monitoring": {"enabled": False}}

        config_path = os.path.join(self.test_config_dir, "disabled_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        collector = MetricsCollector(config_path)

        collector.record_metric("test_metric", 42.5)

        assert "test_metric" not in collector.metrics

    def test_metrics_collector_get_metrics(self):
        """Test getting metrics."""
        collector = MetricsCollector(self.config_path)

        collector.record_metric("test_metric", 42.5)
        collector.record_metric("test_metric", 43.0)

        metrics = collector.get_metrics("test_metric")

        assert len(metrics) == 2
        assert metrics[0]["value"] == 42.5
        assert metrics[1]["value"] == 43.0

    def test_metrics_collector_get_metrics_nonexistent(self):
        """Test getting metrics for non-existent metric."""
        collector = MetricsCollector(self.config_path)

        metrics = collector.get_metrics("nonexistent_metric")

        assert metrics == []

    def test_metrics_collector_get_metrics_with_time_range(self):
        """Test getting metrics within time range."""
        collector = MetricsCollector(self.config_path)

        # Record metrics with specific timestamps
        now = datetime.now()
        old_time = now - timedelta(hours=2)
        recent_time = now - timedelta(minutes=30)

        collector.record_metric("test_metric", 42.5)
        collector.metrics["test_metric"][0]["timestamp"] = old_time

        collector.record_metric("test_metric", 43.0)
        collector.metrics["test_metric"][1]["timestamp"] = recent_time

        # Get metrics from last hour
        start_time = now - timedelta(hours=1)
        metrics = collector.get_metrics("test_metric", start_time=start_time)

        assert len(metrics) == 1
        assert metrics[0]["value"] == 43.0

    def test_metrics_collector_clear_metrics(self):
        """Test clearing metrics."""
        collector = MetricsCollector(self.config_path)

        collector.record_metric("test_metric", 42.5)
        collector.clear_metrics("test_metric")

        assert "test_metric" not in collector.metrics

    def test_metrics_collector_clear_all_metrics(self):
        """Test clearing all metrics."""
        collector = MetricsCollector(self.config_path)

        collector.record_metric("metric1", 42.5)
        collector.record_metric("metric2", 43.0)

        collector.clear_all_metrics()

        assert collector.metrics == {}

    def test_metrics_collector_cleanup_old_metrics(self):
        """Test cleanup of old metrics."""
        collector = MetricsCollector(self.config_path)
        collector.retention_days = 1  # 1 day for testing

        # Record old metric
        collector.record_metric("old_metric", 42.5)
        old_time = datetime.now() - timedelta(days=2)
        collector.metrics["old_metric"][0]["timestamp"] = old_time

        # Record recent metric
        collector.record_metric("recent_metric", 43.0)

        collector._cleanup_old_metrics()

        assert "old_metric" not in collector.metrics
        assert "recent_metric" in collector.metrics


class TestAlertManager:
    """Comprehensive tests for AlertManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_config_dir, "test_config.yaml")

        # Create test config
        test_config = {
            "monitoring": {
                "alerts": {"enabled": True, "email_enabled": False, "webhook_enabled": False}
            }
        }

        with open(self.config_path, "w") as f:
            yaml.dump(test_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_config_dir, ignore_errors=True)

    def test_alert_manager_initialization(self):
        """Test AlertManager initialization."""
        manager = AlertManager(self.config_path)

        assert manager.enabled is True
        assert manager.alerts == []

    def test_alert_manager_initialization_disabled(self):
        """Test AlertManager initialization when disabled."""
        config = {"monitoring": {"alerts": {"enabled": False}}}

        config_path = os.path.join(self.test_config_dir, "disabled_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        manager = AlertManager(config_path)

        assert manager.enabled is False

    def test_alert_manager_trigger_alert(self):
        """Test triggering an alert."""
        manager = AlertManager(self.config_path)

        manager.trigger_alert("error", "Test alert", {"context": "test"})

        assert len(manager.alerts) == 1
        alert = manager.alerts[0]
        assert alert["severity"] == "error"
        assert alert["message"] == "Test alert"
        assert alert["context"] == {"context": "test"}

    def test_alert_manager_trigger_alert_disabled(self):
        """Test triggering alert when disabled."""
        config = {"monitoring": {"alerts": {"enabled": False}}}

        config_path = os.path.join(self.test_config_dir, "disabled_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        manager = AlertManager(config_path)

        manager.trigger_alert("error", "Test alert")

        assert len(manager.alerts) == 0

    def test_alert_manager_get_alerts(self):
        """Test getting alerts."""
        manager = AlertManager(self.config_path)

        manager.trigger_alert("error", "Error alert")
        manager.trigger_alert("warning", "Warning alert")

        alerts = manager.get_alerts()

        assert len(alerts) == 2

    def test_alert_manager_get_alerts_by_severity(self):
        """Test getting alerts by severity."""
        manager = AlertManager(self.config_path)

        manager.trigger_alert("error", "Error alert")
        manager.trigger_alert("warning", "Warning alert")
        manager.trigger_alert("error", "Another error")

        error_alerts = manager.get_alerts(severity="error")

        assert len(error_alerts) == 2

    def test_alert_manager_clear_alerts(self):
        """Test clearing alerts."""
        manager = AlertManager(self.config_path)

        manager.trigger_alert("error", "Error alert")
        manager.clear_alerts()

        assert len(manager.alerts) == 0

    def test_alert_manager_clear_alerts_by_severity(self):
        """Test clearing alerts by severity."""
        manager = AlertManager(self.config_path)

        manager.trigger_alert("error", "Error alert")
        manager.trigger_alert("warning", "Warning alert")

        manager.clear_alerts(severity="error")

        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "warning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
