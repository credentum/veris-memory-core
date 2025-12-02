#!/usr/bin/env python3
"""
Tests for S9 Graph Intent Deprecation (Phase 2).

Validates that S9 returns early with deprecation notice:
- Returns status='pass'
- Returns latency_ms=0
- Returns deprecated=True in details
- Consolidated into S2
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from src.monitoring.sentinel.checks.s9_graph_intent import GraphIntentValidation
from src.monitoring.sentinel.models import SentinelConfig


@pytest.fixture
def mock_config():
    """Create mock sentinel config."""
    config = Mock(spec=SentinelConfig)
    config.target_base_url = "http://localhost:8000"
    config.get = Mock(return_value=None)
    return config


@pytest.fixture
def s9_check(mock_config):
    """Create GraphIntentValidation check instance."""
    return GraphIntentValidation(mock_config)


class TestS9Deprecation:
    """Tests validating S9 deprecation (Phase 2)."""

    @pytest.mark.asyncio
    async def test_returns_early_with_pass_status(self, s9_check):
        """Test that S9 returns immediately with pass status."""
        result = await s9_check.run_check()

        assert result.status == "pass", \
            "Deprecated check should return 'pass' to avoid triggering alerts"

    @pytest.mark.asyncio
    async def test_returns_zero_latency(self, s9_check):
        """Test that S9 returns 0 latency (not executing tests)."""
        result = await s9_check.run_check()

        assert result.latency_ms == 0.0, \
            "Deprecated check should return 0 latency (early return)"

    @pytest.mark.asyncio
    async def test_details_contains_deprecated_flag(self, s9_check):
        """Test that details contains deprecated=True."""
        result = await s9_check.run_check()

        assert "deprecated" in result.details, \
            "Result details should contain 'deprecated' key"
        assert result.details["deprecated"] is True, \
            "deprecated flag should be True"

    @pytest.mark.asyncio
    async def test_details_contains_deprecation_reason(self, s9_check):
        """Test that details explain why deprecated."""
        result = await s9_check.run_check()

        assert "reason" in result.details, \
            "Result details should contain 'reason' key"
        assert "S2" in result.details["reason"], \
            "Reason should mention consolidation into S2"

    @pytest.mark.asyncio
    async def test_details_contains_optimization_info(self, s9_check):
        """Test that details contain optimization metrics."""
        result = await s9_check.run_check()

        assert "optimization" in result.details, \
            "Result details should contain 'optimization' key"
        assert "8" in str(result.details["optimization"]) or "75%" in str(result.details["optimization"]), \
            "Optimization should mention 8 queries or 75% reduction"

    @pytest.mark.asyncio
    async def test_details_contains_cicd_recommendation(self, s9_check):
        """Test that details recommend moving to CI/CD."""
        result = await s9_check.run_check()

        assert "recommendation" in result.details, \
            "Result details should contain 'recommendation' key"
        assert "CI/CD" in result.details["recommendation"], \
            "Recommendation should mention CI/CD pipeline"

    @pytest.mark.asyncio
    async def test_details_contains_consolidated_into_field(self, s9_check):
        """Test that details specify which check absorbed functionality."""
        result = await s9_check.run_check()

        assert "consolidated_into" in result.details, \
            "Result details should contain 'consolidated_into' key"
        assert result.details["consolidated_into"] == "S2-golden-fact-recall", \
            "Should specify consolidation into S2"

    @pytest.mark.asyncio
    async def test_details_contains_phase_info(self, s9_check):
        """Test that details specify optimization phase."""
        result = await s9_check.run_check()

        assert "phase" in result.details, \
            "Result details should contain 'phase' key"
        assert result.details["phase"] == "2" or result.details["phase"] == 2, \
            "Should specify Phase 2 optimization"

    @pytest.mark.asyncio
    async def test_message_contains_deprecation_notice(self, s9_check):
        """Test that message clearly states deprecation."""
        result = await s9_check.run_check()

        assert "DEPRECATED" in result.message.upper(), \
            "Message should contain 'DEPRECATED' (case-insensitive)"
        assert "S2" in result.message, \
            "Message should mention S2 consolidation"

    @pytest.mark.asyncio
    async def test_check_id_unchanged(self, s9_check):
        """Test that check ID remains S9-graph-intent for compatibility."""
        result = await s9_check.run_check()

        assert result.check_id == "S9-graph-intent", \
            "Check ID should remain unchanged for backward compatibility"

    @pytest.mark.asyncio
    async def test_timestamp_is_current(self, s9_check):
        """Test that timestamp is current (not default/zero)."""
        before = datetime.utcnow()
        result = await s9_check.run_check()
        after = datetime.utcnow()

        assert before <= result.timestamp <= after, \
            "Timestamp should be current time"

    @pytest.mark.asyncio
    async def test_does_not_execute_original_tests(self, s9_check):
        """Test that original graph intent tests are NOT executed."""
        # If original tests were executed, latency would be > 0
        # and it would take measurable time
        import time
        start = time.time()
        result = await s9_check.run_check()
        duration = time.time() - start

        assert duration < 0.1, \
            "Check should return immediately (<100ms), not execute full tests"
        assert result.latency_ms == 0.0, \
            "Reported latency should be 0 (early return)"


class TestS9BackwardCompatibility:
    """Tests ensuring S9 deprecation doesn't break monitoring systems."""

    @pytest.mark.asyncio
    async def test_returns_valid_check_result_object(self, s9_check):
        """Test that return value is valid CheckResult."""
        result = await s9_check.run_check()

        # Should have all required CheckResult fields
        assert hasattr(result, "check_id")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "status")
        assert hasattr(result, "latency_ms")
        assert hasattr(result, "message")
        assert hasattr(result, "details")

    @pytest.mark.asyncio
    async def test_status_is_valid_enum_value(self, s9_check):
        """Test that status is a valid value."""
        result = await s9_check.run_check()

        valid_statuses = ["pass", "warn", "fail"]
        assert result.status in valid_statuses, \
            f"Status should be one of {valid_statuses}, got {result.status}"

    @pytest.mark.asyncio
    async def test_details_is_valid_dict(self, s9_check):
        """Test that details is a valid dictionary."""
        result = await s9_check.run_check()

        assert isinstance(result.details, dict), \
            "Details should be a dictionary"
        assert len(result.details) > 0, \
            "Details should not be empty"
