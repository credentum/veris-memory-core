"""
Test utilities for context-store testing.

This module provides common utilities, helpers, and patterns used across test files.
"""

import asyncio
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock


class AsyncTimeout:
    """Async context manager for timeout control in tests."""

    def __init__(self, timeout: float):
        self.timeout = timeout

    async def __aenter__(self):
        self.task = asyncio.current_task()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is asyncio.TimeoutError:
            if self.task:
                self.task.cancel()


@asynccontextmanager
async def async_timeout(timeout: float):
    """Async timeout context manager to prevent hanging tests."""
    try:
        async with asyncio.timeout(timeout):
            yield
    except asyncio.TimeoutError:
        raise AssertionError(f"Test timed out after {timeout} seconds")


@contextmanager
def temporary_config_file(config_data: Dict[str, Any], suffix: str = ".yaml"):
    """Create a temporary configuration file for testing."""
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        yaml.dump(config_data, tmp, default_flow_style=False)
        tmp.flush()

        try:
            yield tmp.name
        finally:
            Path(tmp.name).unlink(missing_ok=True)


class MockResponse:
    """Mock response object for HTTP/API testing."""

    def __init__(self, data: Dict[str, Any], status_code: int = 200):
        self.data = data
        self.status_code = status_code

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class DatabaseTestHelper:
    """Helper class for database testing scenarios."""

    @staticmethod
    def create_neo4j_mock_result(records: List[Dict[str, Any]] = None):
        """Create a mock Neo4j result object."""
        if records is None:
            records = []

        mock_result = Mock()
        mock_result.data.return_value = records
        mock_result.single.return_value = records[0] if records else None

        # Mock record objects
        mock_records = []
        for record_data in records:
            mock_record = Mock()
            for key, value in record_data.items():
                setattr(mock_record, key, value)
            mock_record.data.return_value = record_data
            mock_records.append(mock_record)

        mock_result.__iter__ = lambda: iter(mock_records)
        return mock_result

    @staticmethod
    def create_qdrant_mock_points(count: int = 5):
        """Create mock Qdrant points for testing."""
        from qdrant_client.models import PointStruct

        points = []
        for i in range(count):
            point = Mock(spec=PointStruct)
            point.id = f"point_{i}"
            point.vector = [0.1 * j for j in range(384)]  # Mock 384-dim vector
            point.payload = {"content": f"test content {i}", "metadata": {"index": i}}
            points.append(point)

        return points

    @staticmethod
    def create_redis_mock_data(keys: List[str] = None):
        """Create mock Redis data for testing."""
        if keys is None:
            keys = ["test:key1", "test:key2", "test:key3"]

        mock_data = {}
        for i, key in enumerate(keys):
            mock_data[key] = f"value_{i}"

        return mock_data


class PerformanceTestHelper:
    """Helper class for performance and load testing."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        """Start performance timer."""
        self.start_time = time.time()

    def stop_timer(self):
        """Stop performance timer."""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer not properly started/stopped")
        return self.end_time - self.start_time

    def assert_performance(self, max_seconds: float, operation_name: str = "operation"):
        """Assert that operation completed within time limit."""
        duration = self.get_duration()
        assert (
            duration < max_seconds
        ), f"{operation_name} took {duration:.2f}s, expected < {max_seconds}s"


class ConcurrencyTestHelper:
    """Helper class for concurrency and thread safety testing."""

    @staticmethod
    async def run_concurrent_tasks(coro_func, args_list: List[tuple], max_concurrency: int = 10):
        """Run coroutines concurrently with controlled concurrency."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_task(args):
            async with semaphore:
                return await coro_func(*args)

        tasks = [bounded_task(args) for args in args_list]
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def run_threaded_tasks(func, args_list: List[tuple], max_workers: int = 10):
        """Run functions in multiple threads for thread safety testing."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *args) for args in args_list]
            return [future.result() for future in concurrent.futures.as_completed(futures)]


class ValidationHelper:
    """Helper class for test validation and assertions."""

    @staticmethod
    def assert_valid_uuid(uuid_string: str):
        """Assert that string is a valid UUID."""
        import uuid

        try:
            uuid.UUID(uuid_string)
        except ValueError:
            assert False, f"'{uuid_string}' is not a valid UUID"

    @staticmethod
    def assert_valid_timestamp(timestamp: str):
        """Assert that string is a valid ISO timestamp."""
        from datetime import datetime

        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            assert False, f"'{timestamp}' is not a valid ISO timestamp"

    @staticmethod
    def assert_vector_similarity(
        vector1: List[float], vector2: List[float], tolerance: float = 0.1
    ):
        """Assert that two vectors are similar within tolerance."""
        assert len(vector1) == len(vector2), "Vectors must have same length"

        differences = [abs(v1 - v2) for v1, v2 in zip(vector1, vector2)]
        max_diff = max(differences)

        assert max_diff <= tolerance, f"Vectors differ by {max_diff}, expected <= {tolerance}"

    @staticmethod
    def assert_config_structure(config: Dict[str, Any], required_keys: List[str]):
        """Assert that configuration has required structure."""
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"

        # Check for nested keys (dot notation)
        for key in required_keys:
            if "." in key:
                parts = key.split(".")
                current = config
                for part in parts:
                    assert part in current, f"Missing nested config key: {key}"
                    current = current[part]


# Common test data generators
def generate_test_context_data(count: int = 5) -> List[Dict[str, Any]]:
    """Generate test context data for testing."""
    contexts = []
    for i in range(count):
        contexts.append(
            {
                "id": f"context_{i}",
                "type": "test",
                "content": f"Test content {i}",
                "metadata": {
                    "index": i,
                    "timestamp": "2023-01-01T12:00:00Z",
                    "author": "test_user",
                },
                "embedding": [0.1 * j for j in range(384)],
            }
        )
    return contexts


def generate_test_graph_data(node_count: int = 5) -> Dict[str, Any]:
    """Generate test graph data for Neo4j testing."""
    nodes = []
    relationships = []

    for i in range(node_count):
        nodes.append(
            {
                "id": f"node_{i}",
                "labels": ["TestNode"],
                "properties": {
                    "name": f"Test Node {i}",
                    "index": i,
                    "created": "2023-01-01T12:00:00Z",
                },
            }
        )

        if i > 0:
            relationships.append(
                {
                    "type": "RELATES_TO",
                    "start": f"node_{i-1}",
                    "end": f"node_{i}",
                    "properties": {"weight": 0.5},
                }
            )

    return {"nodes": nodes, "relationships": relationships}


# Test markers for different test types
def integration_test(func):
    """Mark function as integration test."""
    import pytest

    return pytest.mark.integration(func)


def performance_test(func):
    """Mark function as performance test."""
    import pytest

    return pytest.mark.performance(func)


def slow_test(func):
    """Mark function as slow test."""
    import pytest

    return pytest.mark.slow(func)
