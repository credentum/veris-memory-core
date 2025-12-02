#!/usr/bin/env python3
"""
Integration tests that run actual sentinel checks against live services.

PR #402: The CI/CD workflow was showing green but running mocked tests that
didn't actually validate the services. These tests run the REAL checks
against the live Neo4j, Qdrant, Redis, and MCP server started by CI/CD.

Requires:
- Neo4j running on bolt://localhost:7687
- Qdrant running on http://localhost:6333
- Redis running on redis://localhost:6379
- MCP server running on http://localhost:8000

Usage in CI/CD:
    pytest tests/integration/test_sentinel_checks_live.py -v -m integration

Expected behavior:
- Tests should take 10-30 seconds (not <1 second like mocked tests)
- "pass" status = check succeeded
- "warn" status = acceptable (usually means empty database in CI)
- "fail" status = real problem that should block deployment
"""

import pytest
import asyncio
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.monitoring.sentinel.checks.s3_paraphrase_robustness import ParaphraseRobustness
from src.monitoring.sentinel.checks.s4_metrics_wiring import MetricsWiring
from src.monitoring.sentinel.checks.s8_capacity_smoke import CapacitySmoke
from src.monitoring.sentinel.checks.s1_health_probes import VerisHealthProbe
from src.monitoring.sentinel.models import SentinelConfig


@pytest.fixture
def config():
    """Create sentinel config pointing to live services."""
    base_url = os.getenv("TARGET_BASE_URL", "http://localhost:8000")
    return SentinelConfig(target_base_url=base_url)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_s1_health_probes_live(config):
    """
    Run actual S1 health probe check against live services.

    This validates that the MCP server is actually running and responding.
    If this fails, all other checks will likely fail too.
    """
    check = VerisHealthProbe(config)
    result = await check.run_check()

    print(f"\nS1 Health Probes Result:")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    print(f"  Latency: {result.latency_ms:.1f}ms")

    # Health probes should pass if services are running
    assert result.status == "pass", f"S1 health probes failed: {result.message}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_s3_paraphrase_robustness_live(config):
    """
    Run actual S3 paraphrase robustness check against live services.

    This validates that semantic search is working correctly with real
    embeddings and vector similarity, not mocked responses.

    Expected:
    - "pass" if semantic search works correctly
    - "warn" if database is empty (acceptable in CI with fresh DB)
    - "fail" if semantic search has real problems
    """
    check = ParaphraseRobustness(config)
    result = await check.run_check()

    print(f"\nS3 Paraphrase Robustness Result:")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    print(f"  Latency: {result.latency_ms:.1f}ms")
    if result.details:
        print(f"  Passed tests: {result.details.get('passed_tests', 0)}")
        print(f"  Failed tests: {result.details.get('failed_tests', 0)}")
        print(f"  Warned tests: {result.details.get('warned_tests', 0)}")

    # Pass or warn is acceptable (warn = empty database in CI)
    assert result.status in ["pass", "warn"], \
        f"S3 paraphrase robustness failed: {result.message}\nDetails: {result.details}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_s4_metrics_wiring_live(config):
    """
    Run actual S4 metrics wiring check against live services.

    This validates that the metrics endpoint is exposed and returning
    expected metrics format.

    Note: Prometheus/Grafana are optional - check runs in minimal mode
    when they're not available (which is typical in CI).
    """
    check = MetricsWiring(config)
    result = await check.run_check()

    print(f"\nS4 Metrics Wiring Result:")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    print(f"  Latency: {result.latency_ms:.1f}ms")

    # Pass or warn is acceptable (warn = Prometheus/Grafana not available)
    assert result.status in ["pass", "warn"], \
        f"S4 metrics wiring failed: {result.message}\nDetails: {result.details}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_s8_capacity_smoke_live(config):
    """
    Run actual S8 capacity smoke test against live services.

    This validates that the system can handle concurrent requests
    and responds within acceptable latency bounds.

    Note: In CI, this runs a lighter load test than production monitoring.
    """
    check = CapacitySmoke(config)
    result = await check.run_check()

    print(f"\nS8 Capacity Smoke Result:")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    print(f"  Latency: {result.latency_ms:.1f}ms")

    # Pass or warn is acceptable
    assert result.status in ["pass", "warn"], \
        f"S8 capacity smoke failed: {result.message}\nDetails: {result.details}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_checks_complete_within_timeout(config):
    """
    Verify all checks complete within reasonable time.

    This catches any check that hangs or takes unreasonably long,
    which would indicate a problem with service connectivity.
    """
    checks = [
        ("S1", VerisHealthProbe(config)),
        ("S3", ParaphraseRobustness(config)),
        ("S4", MetricsWiring(config)),
        ("S8", CapacitySmoke(config)),
    ]

    results = {}
    total_start = asyncio.get_event_loop().time()

    for check_id, check in checks:
        try:
            # Each check should complete within 60 seconds
            result = await asyncio.wait_for(check.run_check(), timeout=60.0)
            results[check_id] = {
                "status": result.status,
                "latency_ms": result.latency_ms,
                "message": result.message
            }
        except asyncio.TimeoutError:
            results[check_id] = {
                "status": "timeout",
                "message": "Check timed out after 60 seconds"
            }

    total_time = asyncio.get_event_loop().time() - total_start

    print(f"\n{'='*60}")
    print(f"All Checks Summary (total time: {total_time:.1f}s)")
    print(f"{'='*60}")

    for check_id, result in results.items():
        status_emoji = {
            "pass": "✅",
            "warn": "⚠️",
            "fail": "❌",
            "timeout": "⏰"
        }.get(result["status"], "❓")

        latency = result.get("latency_ms", 0)
        print(f"  {status_emoji} {check_id}: {result['status']} ({latency:.0f}ms)")

    # Verify no timeouts occurred
    timeouts = [k for k, v in results.items() if v["status"] == "timeout"]
    assert not timeouts, f"Checks timed out: {timeouts}"

    # Verify no hard failures (warn is acceptable)
    failures = [k for k, v in results.items() if v["status"] == "fail"]
    assert not failures, f"Checks failed: {failures}"
