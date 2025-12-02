"""Pytest configuration for context-store tests.

This file configures the test environment and handles import paths centrally.
All test files should use this configuration - DO NOT add sys.path manipulations
in individual test files.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

# Centralized sys.path configuration for all tests
# This allows tests to import from src/ directly without individual setup
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("CONFIG_PATH", str(project_root / "config" / "test.yaml"))

# Import test configuration utilities
from src.core.test_config import (  # noqa: E402
    get_minimal_config, 
    get_test_config, 
    get_test_credentials,
    get_mock_responses,
    get_alternative_config,
    get_database_url
)


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide full test configuration for storage components."""
    return get_test_config()


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Provide minimal test configuration."""
    return get_minimal_config()


@pytest.fixture
def mock_neo4j_config() -> Dict[str, Any]:
    """Provide Neo4j-specific test configuration."""
    config = get_test_config()
    return {"neo4j": config["neo4j"]}


@pytest.fixture
def mock_qdrant_config() -> Dict[str, Any]:
    """Provide Qdrant-specific test configuration."""
    config = get_test_config()
    return {"qdrant": config["qdrant"]}


@pytest.fixture
def mock_redis_config() -> Dict[str, Any]:
    """Provide Redis-specific test configuration."""
    config = get_test_config()
    return {"redis": config["redis"]}


@pytest.fixture
def mock_config_loader(test_config):
    """Mock the config loader to return test configuration."""
    with patch("storage.neo4j_client.open", side_effect=FileNotFoundError):
        with patch("storage.qdrant_client.open", side_effect=FileNotFoundError):
            with patch("storage.kv_store.open", side_effect=FileNotFoundError):
                yield test_config


@pytest.fixture
def neo4j_client_mock(test_config):
    """Create a mock Neo4j client for testing."""
    from src.storage.neo4j_client import Neo4jInitializer

    client = Neo4jInitializer(config=test_config, test_mode=True)
    return client


@pytest.fixture
def qdrant_client_mock(test_config):
    """Create a mock Qdrant client for testing."""
    from src.storage.qdrant_client import VectorDBInitializer

    client = VectorDBInitializer(config=test_config, test_mode=True)
    return client


@pytest.fixture
def kv_store_mock(test_config):
    """Create a mock KV store for testing."""
    from src.storage.kv_store import ContextKV

    store = ContextKV(config=test_config, test_mode=True)
    return store


# Enhanced Database Mocking Fixtures for Phase 2 & 3


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for comprehensive testing."""
    from unittest.mock import Mock

    # Create mock driver
    mock_driver = Mock()
    mock_session = Mock()
    mock_tx = Mock()

    # Configure mock behavior
    mock_driver.session.return_value = mock_session
    mock_session.begin_transaction.return_value = mock_tx
    mock_session.run.return_value = Mock(data=lambda: [])
    mock_tx.run.return_value = Mock(data=lambda: [])

    # Mock async methods
    mock_session.close = Mock()
    mock_tx.commit = Mock()
    mock_tx.rollback = Mock()

    with patch("neo4j.GraphDatabase.driver", return_value=mock_driver):
        yield mock_driver


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for comprehensive testing."""
    from unittest.mock import Mock

    from qdrant_client.models import Distance, VectorParams

    # Create mock client
    mock_client = Mock()

    # Configure collection operations
    mock_client.get_collections.return_value.collections = []
    mock_client.create_collection = Mock()
    mock_client.delete_collection = Mock()
    mock_client.collection_exists.return_value = False

    # Configure vector operations
    mock_client.upsert = Mock()
    mock_client.search = Mock(return_value=[])
    mock_client.scroll = Mock(return_value=([], None))
    mock_client.delete = Mock()

    # Configure info operations
    mock_client.get_collection.return_value = Mock(
        config=Mock(params=VectorParams(size=384, distance=Distance.COSINE))
    )

    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for comprehensive testing."""
    from unittest.mock import Mock

    # Create mock client
    mock_client = Mock()

    # Configure Redis operations
    mock_client.get = Mock(return_value=None)
    mock_client.set = Mock(return_value=True)
    mock_client.delete = Mock(return_value=1)
    mock_client.exists = Mock(return_value=False)
    mock_client.expire = Mock(return_value=True)
    mock_client.ttl = Mock(return_value=-1)

    # Configure Redis collections
    mock_client.hget = Mock(return_value=None)
    mock_client.hset = Mock(return_value=1)
    mock_client.hgetall = Mock(return_value={})
    mock_client.hdel = Mock(return_value=1)

    # Configure async operations
    mock_client.ping = Mock(return_value=True)

    with patch("redis.Redis", return_value=mock_client):
        yield mock_client


@pytest.fixture
def docker_compose_project_name():
    """Provide project name for Docker Compose tests."""
    return "context-store-test"


@pytest.fixture(scope="session")
def docker_compose_file():
    """Provide Docker Compose file path for integration tests."""
    return "docker-compose.test.yml"


@pytest.fixture
def test_database_config():
    """Provide test database configuration for integration tests."""
    config = get_alternative_config("integration_test")
    return {
        "neo4j": {
            "uri": get_database_url("neo4j"),
            "username": config["neo4j"]["username"],
            "password": get_test_credentials("neo4j").get("test_pass", "testpassword"),
            "database": config["neo4j"]["database"],
        },
        "qdrant": {
            "url": get_database_url("qdrant"),
            "collection_name": config["qdrant"]["collection_name"],
            "vector_size": config["qdrant"]["dimensions"],
        },
        "redis": {
            "url": get_database_url("redis"),
            "prefix": "test:",
        },
    }


@pytest.fixture
def async_test_timeout():
    """Provide timeout for async tests to prevent hanging."""
    return 30.0  # 30 seconds


@pytest.fixture
def embedding_mock():
    """Mock embedding service for testing."""
    from unittest.mock import Mock

    mock_service = Mock()
    mock_service.embed_text.return_value = [0.1] * 384  # Mock 384-dim vector
    mock_service.embed_batch.return_value = [[0.1] * 384] * 5  # Mock batch

    return mock_service


@pytest.fixture
def rate_limiter_mock():
    """Mock rate limiter for testing."""
    from unittest.mock import Mock

    mock_limiter = Mock()
    mock_limiter.check_rate_limit.return_value = (True, None)  # Allow all requests
    mock_limiter.get_rate_limit_info.return_value = {"status": "ok"}

    return mock_limiter


@pytest.fixture(autouse=True)
def isolated_prometheus_registry():
    """Provide isolated Prometheus registry for each test to avoid global state interference."""
    try:
        from prometheus_client import CollectorRegistry, REGISTRY
        import threading
        
        # Create an isolated registry for this test
        test_registry = CollectorRegistry()
        
        # Store original registry reference
        original_registry = REGISTRY
        
        # Temporarily replace the global registry with our isolated one
        # We need to be careful about thread-local state
        prometheus_client_module = __import__('prometheus_client')
        original_registry_ref = prometheus_client_module.REGISTRY
        prometheus_client_module.REGISTRY = test_registry
        
        # Also handle any existing collectors that might reference the old registry
        # This prevents cross-test contamination
        yield test_registry
        
        # Restore original registry
        prometheus_client_module.REGISTRY = original_registry_ref
        
    except ImportError:
        # Prometheus not available, yield None
        yield None


@pytest.fixture
def prometheus_registry():
    """Provide a fresh Prometheus registry for tests that explicitly need metrics.
    
    This fixture creates a completely isolated registry that doesn't interfere 
    with the global state or other tests.
    """
    try:
        from prometheus_client import CollectorRegistry
        
        # Create a fresh, isolated registry
        registry = CollectorRegistry()
        return registry
        
    except ImportError:
        pytest.skip("Prometheus client not available")


@pytest.fixture  
def mock_metrics_with_registry(prometheus_registry):
    """Create mock metrics objects bound to an isolated registry."""
    try:
        from prometheus_client import Counter, Histogram, Gauge
        
        # Create metrics with the isolated registry
        metrics = {
            'request_counter': Counter(
                'test_requests_total', 
                'Test requests', 
                ['endpoint'], 
                registry=prometheus_registry
            ),
            'request_duration': Histogram(
                'test_request_duration_seconds',
                'Test request duration',
                ['endpoint'],
                registry=prometheus_registry
            ),
            'active_connections': Gauge(
                'test_active_connections',
                'Test active connections',
                registry=prometheus_registry
            )
        }
        
        return metrics, prometheus_registry
        
    except ImportError:
        pytest.skip("Prometheus client not available")


@pytest.fixture
def test_credentials():
    """Provide test credentials from centralized configuration."""
    return get_test_credentials()


@pytest.fixture  
def mock_service_responses():
    """Provide mock service responses from centralized configuration."""
    return get_mock_responses()


@pytest.fixture
def sample_test_data():
    """Provide sample test data from centralized configuration."""
    from src.core.test_config import get_test_data
    return get_test_data()


@pytest.fixture
def integration_test_config():
    """Provide integration test configuration."""
    return get_alternative_config("integration_test")


@pytest.fixture
def ssl_test_config():
    """Provide SSL test configuration."""
    return get_alternative_config("ssl_test")
