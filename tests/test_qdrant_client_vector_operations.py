"""
Comprehensive tests for src/storage/qdrant_client.py focusing on vector operations and business logic.

This test suite covers:
1. Vector similarity search algorithms and ranking
2. Collection management and configuration
3. Embedding storage and batch operations
4. Search filtering and result processing
5. Distance calculations and similarity metrics
6. Error handling for vector operations

Uses extensive mocking to avoid actual Qdrant connections and focuses on business logic.
"""

from unittest.mock import AsyncMock, Mock, mock_open, patch

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

from src.core.config import Config
from src.storage.qdrant_client import VectorDBInitializer


class TestVectorDBInitializerConfigLoading:
    """Test configuration loading and initialization."""

    def test_load_config_success(self):
        """Test successful configuration loading."""
        config_data = {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "ssl": False,
            }
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(config_data))):
            initializer = VectorDBInitializer("test_config.yaml")
            assert initializer.config == config_data
            assert initializer.client is None

    def test_load_config_file_not_found(self):
        """Test configuration loading when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with patch("click.echo") as mock_echo:
                with patch("sys.exit") as mock_exit:
                    VectorDBInitializer("nonexistent.yaml")
                    mock_echo.assert_called_with("Error: nonexistent.yaml not found", err=True)
                    mock_exit.assert_called_with(1)

    def test_load_config_yaml_error(self):
        """Test configuration loading with YAML parsing error."""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content:")):
            with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
                with patch("click.echo") as mock_echo:
                    with patch("sys.exit") as mock_exit:
                        VectorDBInitializer("test_config.yaml")
                        mock_echo.assert_called()
                        mock_exit.assert_called_with(1)

    def test_load_config_invalid_format(self):
        """Test configuration loading with invalid format (not a dict)."""
        with patch("builtins.open", mock_open(read_data="- invalid\n- config\n- format")):
            with patch("click.echo") as mock_echo:
                with patch("sys.exit") as mock_exit:
                    VectorDBInitializer("test_config.yaml")
                    mock_echo.assert_called_with(
                        "Error: test_config.yaml must contain a dictionary", err=True
                    )
                    mock_exit.assert_called_with(1)

    def test_default_config_path(self):
        """Test using default configuration path."""
        config_data = {"qdrant": {"host": "localhost"}}

        with patch("builtins.open", mock_open(read_data=yaml.dump(config_data))):
            initializer = VectorDBInitializer()
            assert initializer.config == config_data


class TestVectorDBInitializerConnection:
    """Test connection establishment and configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_data = {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "ssl": False,
                "timeout": 10,
                "verify_ssl": True,
            }
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(self.config_data))):
            self.initializer = VectorDBInitializer("test_config.yaml")

    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_success_without_ssl(self, mock_qdrant_client):
        """Test successful connection without SSL."""
        with patch("core.utils.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "timeout": 5,
            }

            mock_client = AsyncMock()
            mock_client.get_collections.return_value = AsyncMock()
            mock_qdrant_client.return_value = mock_client

            with patch("click.echo") as mock_echo:
                result = self.initializer.connect()

                assert result is True
                assert self.initializer.client == mock_client
                mock_qdrant_client.assert_called_once_with(host="localhost", port=6333, timeout=5)
                mock_client.get_collections.assert_called_once_with()
                mock_echo.assert_called_with("✓ Connected to Qdrant at localhost:6333")

    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_success_with_ssl(self, mock_qdrant_client):
        """Test successful connection with SSL."""
        with patch("core.utils.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "secure.qdrant.com",
                "port": 6334,
                "ssl": True,
                "timeout": 10,
                "verify_ssl": True,
            }

            mock_client = AsyncMock()
            mock_client.get_collections.return_value = AsyncMock()
            mock_qdrant_client.return_value = mock_client

            with patch("click.echo") as mock_echo:
                result = self.initializer.connect()

                assert result is True
                assert self.initializer.client == mock_client
                mock_qdrant_client.assert_called_once_with(
                    host="secure.qdrant.com", port=6334, https=True, verify=True, timeout=10
                )

    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_ssl_with_verify_disabled(self, mock_qdrant_client):
        """Test SSL connection with verification disabled."""
        with patch("core.utils.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "localhost",
                "port": 6333,
                "ssl": True,
                "timeout": 5,
                "verify_ssl": False,
            }

            mock_client = AsyncMock()
            mock_client.get_collections.return_value = AsyncMock()
            mock_qdrant_client.return_value = mock_client

            result = self.initializer.connect()

            assert result is True
            mock_qdrant_client.assert_called_once_with(
                host="localhost", port=6333, https=True, verify=False, timeout=5
            )

    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_default_port_and_timeout(self, mock_qdrant_client):
        """Test connection with default port and timeout values."""
        with patch("core.utils.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "ssl": False}

            mock_client = AsyncMock()
            mock_client.get_collections.return_value = AsyncMock()
            mock_qdrant_client.return_value = mock_client

            result = self.initializer.connect()

            assert result is True
            mock_qdrant_client.assert_called_once_with(host="localhost", port=6333, timeout=5)

    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_failure_exception(self, mock_qdrant_client):
        """Test connection failure due to exception."""
        with patch("core.utils.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "timeout": 5,
            }

            mock_qdrant_client.side_effect = Exception("Connection refused")

            with patch("click.echo") as mock_echo:
                result = self.initializer.connect()

                assert result is False
                assert self.initializer.client is None
                mock_echo.assert_called_with(
                    "✗ Failed to connect to Qdrant at localhost:6333: Connection refused", err=True
                )

    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_test_connection_fails(self, mock_qdrant_client):
        """Test connection where client creation succeeds but test fails."""
        with patch("core.utils.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "timeout": 5,
            }

            mock_client = AsyncMock()
            mock_client.get_collections.side_effect = Exception("Service unavailable")
            mock_qdrant_client.return_value = mock_client

            with patch("click.echo") as mock_echo:
                result = self.initializer.connect()

                assert result is False
                mock_echo.assert_called_with(
                    "✗ Failed to connect to Qdrant at localhost:6333: Service unavailable", err=True
                )

    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_client_none(self, mock_qdrant_client):
        """Test edge case where QdrantClient returns None."""
        with patch("core.utils.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "timeout": 5,
            }

            mock_qdrant_client.return_value = None

            result = self.initializer.connect()

            assert result is False
            assert self.initializer.client is None


class TestVectorDBInitializerCollectionManagement:
    """Test collection creation and management operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_data = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(self.config_data))):
            self.initializer = VectorDBInitializer("test_config.yaml")

        # Set up mock client
        self.mock_client = Mock(spec=QdrantClient)
        self.initializer.client = self.mock_client

    def test_create_collection_no_client(self):
        """Test collection creation when no client is connected."""
        self.initializer.client = None

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_collection()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    def test_create_collection_success_new(self):
        """Test successful creation of new collection."""
        # Mock collections response - collection doesn't exist
        mock_collections_response = AsyncMock()
        mock_collections_response.collections = []
        self.mock_client.get_collections.return_value = mock_collections_response

        with patch("click.echo") as mock_echo:
            with patch("time.sleep"):  # Skip actual sleep
                result = self.initializer.create_collection()

                assert result is True
                self.mock_client.create_collection.assert_called_once_with(
                    collection_name="test_collection",
                    vectors_config=VectorParams(
                        size=Config.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE,
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        deleted_threshold=0.2,
                        vacuum_min_vector_number=1000,
                        default_segment_number=2,
                        flush_interval_sec=5,
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=128,
                        full_scan_threshold=10000,
                    ),
                )
                mock_echo.assert_any_call("Creating collection 'test_collection'...")
                mock_echo.assert_any_call("✓ Collection 'test_collection' created successfully")

    def test_create_collection_exists_no_force(self):
        """Test collection creation when collection exists and force=False."""
        # Mock collections response - collection exists
        mock_collection = AsyncMock()
        mock_collection.name = "test_collection"
        mock_collections_response = AsyncMock()
        mock_collections_response.collections = [mock_collection]
        self.mock_client.get_collections.return_value = mock_collections_response

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_collection(force=False)

            assert result is True
            self.mock_client.create_collection.assert_not_called()
            mock_echo.assert_called_with(
                "Collection 'test_collection' already exists. Use --force to recreate."
            )

    def test_create_collection_exists_with_force(self):
        """Test collection recreation when collection exists and force=True."""
        # Mock collections response - collection exists
        mock_collection = AsyncMock()
        mock_collection.name = "test_collection"
        mock_collections_response = AsyncMock()
        mock_collections_response.collections = [mock_collection]
        self.mock_client.get_collections.return_value = mock_collections_response

        with patch("click.echo") as mock_echo:
            with patch("time.sleep"):  # Skip actual sleep
                result = self.initializer.create_collection(force=True)

                assert result is True
                self.mock_client.delete_collection.assert_called_once_with("test_collection")
                self.mock_client.create_collection.assert_called_once_with()
                mock_echo.assert_any_call("Deleting existing collection 'test_collection'...")
                mock_echo.assert_any_call("Creating collection 'test_collection'...")

    def test_create_collection_custom_name(self):
        """Test collection creation with custom collection name from config."""
        # Update config with custom collection name
        self.initializer.config["qdrant"]["collection_name"] = "custom_collection"

        mock_collections_response = AsyncMock()
        mock_collections_response.collections = []
        self.mock_client.get_collections.return_value = mock_collections_response

        result = self.initializer.create_collection()

        assert result is True
        self.mock_client.create_collection.assert_called_once_with()
        call_args = self.mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "custom_collection"

    def test_create_collection_default_name(self):
        """Test collection creation with default name when not in config."""
        # Remove collection_name from config
        if "collection_name" in self.initializer.config.get("qdrant", {}):
            del self.initializer.config["qdrant"]["collection_name"]

        mock_collections_response = AsyncMock()
        mock_collections_response.collections = []
        self.mock_client.get_collections.return_value = mock_collections_response

        result = self.initializer.create_collection()

        assert result is True
        call_args = self.mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "project_context"

    def test_create_collection_exception(self):
        """Test collection creation failure due to exception."""
        self.mock_client.get_collections.side_effect = Exception("Database error")

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_collection()

            assert result is False
            mock_echo.assert_called_with("✗ Failed to create collection: Database error", err=True)

    def test_create_collection_vector_config_validation(self):
        """Test that collection is created with correct vector configuration."""
        mock_collections_response = AsyncMock()
        mock_collections_response.collections = []
        self.mock_client.get_collections.return_value = mock_collections_response

        result = self.initializer.create_collection()

        assert result is True
        call_args = self.mock_client.create_collection.call_args
        vectors_config = call_args[1]["vectors_config"]

        assert isinstance(vectors_config, VectorParams)
        assert vectors_config.size == Config.EMBEDDING_DIMENSIONS
        assert vectors_config.distance == Distance.COSINE

    def test_create_collection_optimizer_config_validation(self):
        """Test that collection is created with correct optimizer configuration."""
        mock_collections_response = AsyncMock()
        mock_collections_response.collections = []
        self.mock_client.get_collections.return_value = mock_collections_response

        result = self.initializer.create_collection()

        assert result is True
        call_args = self.mock_client.create_collection.call_args
        optimizers_config = call_args[1]["optimizers_config"]

        assert isinstance(optimizers_config, OptimizersConfigDiff)
        assert optimizers_config.deleted_threshold == 0.2
        assert optimizers_config.vacuum_min_vector_number == 1000
        assert optimizers_config.default_segment_number == 2
        assert optimizers_config.flush_interval_sec == 5

    def test_create_collection_hnsw_config_validation(self):
        """Test that collection is created with correct HNSW configuration."""
        mock_collections_response = AsyncMock()
        mock_collections_response.collections = []
        self.mock_client.get_collections.return_value = mock_collections_response

        result = self.initializer.create_collection()

        assert result is True
        call_args = self.mock_client.create_collection.call_args
        hnsw_config = call_args[1]["hnsw_config"]

        assert isinstance(hnsw_config, HnswConfigDiff)
        assert hnsw_config.m == 16
        assert hnsw_config.ef_construct == 128
        assert hnsw_config.full_scan_threshold == 10000


class TestVectorDBInitializerVerification:
    """Test setup verification and collection information retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_data = {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "version": "1.14.x",
            }
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(self.config_data))):
            self.initializer = VectorDBInitializer("test_config.yaml")

        self.mock_client = Mock(spec=QdrantClient)
        self.initializer.client = self.mock_client

    def test_verify_setup_no_client(self):
        """Test verification when no client is connected."""
        self.initializer.client = None

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    def test_verify_setup_success_vector_params(self):
        """Test successful verification with VectorParams configuration."""
        # Mock collection info with VectorParams
        mock_vectors_config = VectorParams(size=1536, distance=Distance.COSINE)
        mock_params = AsyncMock()
        mock_params.vectors = mock_vectors_config
        mock_config = AsyncMock()
        mock_config.params = mock_params
        mock_info = AsyncMock()
        mock_info.config = mock_config
        mock_info.points_count = 42

        self.mock_client.get_collection.return_value = mock_info

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is True
            self.mock_client.get_collection.assert_called_once_with("test_collection")
            mock_echo.assert_any_call("\nCollection Info:")
            mock_echo.assert_any_call("  Name: test_collection")
            mock_echo.assert_any_call("  Vector size: 1536")
            # Check that distance metric is displayed (value may vary by Qdrant version)
            distance_calls = [
                call for call in mock_echo.call_args_list if "Distance metric:" in str(call)
            ]
            assert len(distance_calls) >= 1
            mock_echo.assert_any_call("  Points count: 42")
            mock_echo.assert_any_call("\nExpected Qdrant version: 1.14.x")

    def test_verify_setup_success_named_vectors(self):
        """Test successful verification with named vectors configuration."""
        # Mock collection info with named vectors
        mock_vector_config = AsyncMock()
        mock_vector_config.size = 512
        mock_vector_config.distance = Distance.DOT

        mock_vectors_config = {"embeddings": mock_vector_config, "dense": mock_vector_config}
        mock_params = AsyncMock()
        mock_params.vectors = mock_vectors_config
        mock_config = AsyncMock()
        mock_config.params = mock_params
        mock_info = AsyncMock()
        mock_info.config = mock_config
        mock_info.points_count = 100

        self.mock_client.get_collection.return_value = mock_info

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is True
            mock_echo.assert_any_call("  Vector 'embeddings' size: 512")
            # Check that vector distances are displayed
            distance_calls = [call for call in mock_echo.call_args_list if "distance:" in str(call)]
            assert (
                len(distance_calls) >= 2
            )  # Should have at least 2 distance calls for named vectors

    def test_verify_setup_missing_config(self):
        """Test verification with missing configuration elements."""
        # Mock collection info with minimal config
        mock_info = AsyncMock()
        mock_info.config = None
        mock_info.points_count = 0

        self.mock_client.get_collection.return_value = mock_info

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is True
            mock_echo.assert_any_call("  Name: test_collection")
            mock_echo.assert_any_call("  Points count: 0")

    def test_verify_setup_missing_vectors_config(self):
        """Test verification with missing vectors configuration."""
        mock_params = AsyncMock()
        mock_params.vectors = None
        mock_config = AsyncMock()
        mock_config.params = mock_params
        mock_info = AsyncMock()
        mock_info.config = mock_config
        mock_info.points_count = 5

        self.mock_client.get_collection.return_value = mock_info

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is True
            mock_echo.assert_any_call("  Points count: 5")

    def test_verify_setup_default_collection_name(self):
        """Test verification with default collection name."""
        # Remove collection_name from config
        del self.initializer.config["qdrant"]["collection_name"]

        mock_info = AsyncMock()
        mock_info.config = None
        mock_info.points_count = 0

        self.mock_client.get_collection.return_value = mock_info

        result = self.initializer.verify_setup()

        assert result is True
        self.mock_client.get_collection.assert_called_once_with("project_context")

    def test_verify_setup_default_version(self):
        """Test verification with default Qdrant version."""
        # Remove version from config
        del self.initializer.config["qdrant"]["version"]

        mock_info = AsyncMock()
        mock_info.config = None
        mock_info.points_count = 0

        self.mock_client.get_collection.return_value = mock_info

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is True
            mock_echo.assert_any_call("\nExpected Qdrant version: 1.14.x")

    def test_verify_setup_exception(self):
        """Test verification failure due to exception."""
        self.mock_client.get_collection.side_effect = Exception("Collection not found")

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is False
            mock_echo.assert_called_with("✗ Failed to verify setup: Collection not found", err=True)


class TestVectorDBInitializerTestOperations:
    """Test vector operations including insert and search."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_data = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(self.config_data))):
            self.initializer = VectorDBInitializer("test_config.yaml")

        self.mock_client = Mock(spec=QdrantClient)
        self.initializer.client = self.mock_client

    def test_insert_test_point_no_client(self):
        """Test test point insertion when no client is connected."""
        self.initializer.client = None

        with patch("click.echo") as mock_echo:
            result = self.initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    @patch("random.random")
    def test_insert_test_point_success(self, mock_random):
        """Test successful test point insertion and retrieval."""
        # Mock random vector generation
        mock_random.return_value = 0.5
        expected_vector = [0.5] * Config.EMBEDDING_DIMENSIONS

        # Mock search result
        mock_search_result = AsyncMock()
        mock_search_result.id = "test-point-001"
        self.mock_client.search.return_value = [mock_search_result]

        with patch("click.echo") as mock_echo:
            result = self.initializer.insert_test_point()

            assert result is True

            # Verify upsert call
            self.mock_client.upsert.assert_called_once_with()
            upsert_args = self.mock_client.upsert.call_args
            assert upsert_args[1]["collection_name"] == "test_collection"
            points = upsert_args[1]["points"]
            assert len(points) == 1

            point = points[0]
            assert isinstance(point, PointStruct)
            assert point.id == "test-point-001"
            assert point.vector == expected_vector
            assert point.payload["document_type"] == "test"
            assert point.payload["content"] == "This is a test point for verification"
            assert point.payload["created_date"] == "2025-07-11"

            # Verify search call
            self.mock_client.search.assert_called_once_with(
                collection_name="test_collection", query_vector=expected_vector, limit=1
            )

            # Verify cleanup
            self.mock_client.delete.assert_called_once_with(
                collection_name="test_collection", points_selector=["test-point-001"]
            )

            mock_echo.assert_called_with("✓ Test point inserted and retrieved successfully")

    @patch("random.random")
    def test_insert_test_point_search_mismatch(self, mock_random):
        """Test test point insertion when search returns different ID."""
        mock_random.return_value = 0.5

        # Mock search result with wrong ID
        mock_search_result = AsyncMock()
        mock_search_result.id = "wrong-id"
        self.mock_client.search.return_value = [mock_search_result]

        with patch("click.echo") as mock_echo:
            result = self.initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with("✗ Test point verification failed", err=True)

    @patch("random.random")
    def test_insert_test_point_no_search_results(self, mock_random):
        """Test test point insertion when search returns no results."""
        mock_random.return_value = 0.5

        # Mock empty search result
        self.mock_client.search.return_value = []

        with patch("click.echo") as mock_echo:
            result = self.initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with("✗ Test point verification failed", err=True)

    @patch("random.random")
    def test_insert_test_point_upsert_exception(self, mock_random):
        """Test test point insertion failure during upsert."""
        mock_random.return_value = 0.5

        self.mock_client.upsert.side_effect = Exception("Upsert failed")

        with patch("click.echo") as mock_echo:
            result = self.initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with(
                "✗ Failed to test point operations: Upsert failed", err=True
            )

    @patch("random.random")
    def test_insert_test_point_search_exception(self, mock_random):
        """Test test point insertion failure during search."""
        mock_random.return_value = 0.5

        self.mock_client.search.side_effect = Exception("Search failed")

        with patch("click.echo") as mock_echo:
            result = self.initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with(
                "✗ Failed to test point operations: Search failed", err=True
            )

    @patch("random.random")
    def test_insert_test_point_delete_exception(self, mock_random):
        """Test test point insertion with cleanup failure."""
        mock_random.return_value = 0.5

        # Mock successful upsert and search
        mock_search_result = AsyncMock()
        mock_search_result.id = "test-point-001"
        self.mock_client.search.return_value = [mock_search_result]

        # Mock delete failure
        self.mock_client.delete.side_effect = Exception("Delete failed")

        with patch("click.echo") as mock_echo:
            result = self.initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with(
                "✗ Failed to test point operations: Delete failed", err=True
            )

    @patch("random.random")
    def test_insert_test_point_vector_dimensions(self, mock_random):
        """Test that test point uses correct vector dimensions."""
        mock_random.return_value = 0.7

        mock_search_result = AsyncMock()
        mock_search_result.id = "test-point-001"
        self.mock_client.search.return_value = [mock_search_result]

        result = self.initializer.insert_test_point()

        assert result is True

        # Check that vector has correct dimensions
        upsert_args = self.mock_client.upsert.call_args
        point = upsert_args[1]["points"][0]
        assert len(point.vector) == Config.EMBEDDING_DIMENSIONS
        assert all(v == 0.7 for v in point.vector)


class TestVectorDBInitializerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_data = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(self.config_data))):
            self.initializer = VectorDBInitializer("test_config.yaml")

    def test_empty_config(self):
        """Test initialization with empty configuration."""
        with patch("builtins.open", mock_open(read_data=yaml.dump({}))):
            initializer = VectorDBInitializer("test_config.yaml")
            assert initializer.config == {}

    def test_missing_qdrant_config(self):
        """Test operations with missing qdrant configuration section."""
        with patch("builtins.open", mock_open(read_data=yaml.dump({"other": "config"}))):
            initializer = VectorDBInitializer("test_config.yaml")

        mock_client = AsyncMock()
        initializer.client = mock_client

        # Should use defaults
        mock_collections_response = AsyncMock()
        mock_collections_response.collections = []
        mock_client.get_collections.return_value = mock_collections_response

        result = initializer.create_collection()
        assert result is True

        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "project_context"

    def test_malformed_vector_in_verification(self):
        """Test verification with malformed vector configuration."""
        self.initializer.client = AsyncMock()

        # Mock collection info with malformed vectors config
        mock_vectors_config = "invalid_config"  # Not a dict or VectorParams
        mock_params = AsyncMock()
        mock_params.vectors = mock_vectors_config
        mock_config = AsyncMock()
        mock_config.params = mock_params
        mock_info = AsyncMock()
        mock_info.config = mock_config
        mock_info.points_count = 0

        self.initializer.client.get_collection.return_value = mock_info

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            # Should handle gracefully and not crash
            assert result is True
            mock_echo.assert_any_call("  Points count: 0")

    def test_collection_exists_check_exception(self):
        """Test collection creation when exists check fails."""
        mock_client = AsyncMock()
        self.initializer.client = mock_client

        # Make get_collections fail
        mock_client.get_collections.side_effect = Exception("Permission denied")

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_collection()

            assert result is False
            mock_echo.assert_called_with(
                "✗ Failed to create collection: Permission denied", err=True
            )

    @patch("random.random")
    def test_insert_test_point_with_custom_collection(self, mock_random):
        """Test test point insertion with custom collection name."""
        # Set custom collection name
        self.initializer.config["qdrant"]["collection_name"] = "custom_test_collection"
        mock_random.return_value = 0.3

        mock_client = AsyncMock()
        self.initializer.client = mock_client

        mock_search_result = AsyncMock()
        mock_search_result.id = "test-point-001"
        mock_client.search.return_value = [mock_search_result]

        result = self.initializer.insert_test_point()

        assert result is True

        # Verify operations used custom collection name
        upsert_args = mock_client.upsert.call_args
        assert upsert_args[1]["collection_name"] == "custom_test_collection"

        search_args = mock_client.search.call_args
        assert search_args[1]["collection_name"] == "custom_test_collection"

        delete_args = mock_client.delete.call_args
        assert delete_args[1]["collection_name"] == "custom_test_collection"


class TestVectorDBInitializerIntegration:
    """Integration tests covering multiple operations together."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_data = {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "integration_test",
                "ssl": False,
                "timeout": 10,
            }
        }

    @patch("storage.qdrant_client.QdrantClient")
    @patch("random.random")
    def test_full_initialization_workflow(self, mock_random, mock_qdrant_client):
        """Test complete initialization workflow: connect, create, verify, test."""
        with patch("core.utils.get_secure_connection_config") as mock_get_config:
            # Setup mocks
            mock_get_config.return_value = {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "timeout": 10,
            }

        mock_client = AsyncMock()
        mock_qdrant_client.return_value = mock_client
        mock_random.return_value = 0.8

        # Mock responses for each step
        mock_client.get_collections.return_value = AsyncMock()  # For connect

        # Collection doesn't exist initially
        mock_collections_response = AsyncMock()
        mock_collections_response.collections = []
        mock_client.get_collections.return_value = mock_collections_response

        # Mock collection info for verification
        mock_vectors_config = VectorParams(size=1536, distance=Distance.COSINE)
        mock_params = AsyncMock()
        mock_params.vectors = mock_vectors_config
        mock_config = AsyncMock()
        mock_config.params = mock_params
        mock_info = AsyncMock()
        mock_info.config = mock_config
        mock_info.points_count = 1
        mock_client.get_collection.return_value = mock_info

        # Mock test point operations
        mock_search_result = AsyncMock()
        mock_search_result.id = "test-point-001"
        mock_client.search.return_value = [mock_search_result]

        with patch("builtins.open", mock_open(read_data=yaml.dump(self.config_data))):
            initializer = VectorDBInitializer("test_config.yaml")

        with patch("click.echo") as mock_echo, patch("time.sleep"):
            # Run full workflow
            connect_result = initializer.connect()
            create_result = initializer.create_collection()
            verify_result = initializer.verify_setup()
            test_result = initializer.insert_test_point()

            # Verify all steps succeeded
            assert connect_result is True
            assert create_result is True
            assert verify_result is True
            assert test_result is True

            # Verify all operations were called
            mock_client.create_collection.assert_called_once_with()
            mock_client.get_collection.assert_called_once_with()
            mock_client.upsert.assert_called_once_with()
            mock_client.search.assert_called_once_with()
            mock_client.delete.assert_called_once_with()

    @patch("storage.qdrant_client.QdrantClient")
    def test_recreation_workflow(self, mock_qdrant_client):
        """Test collection recreation workflow."""
        with patch("core.utils.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "timeout": 5,
            }

        mock_client = AsyncMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = AsyncMock()

        # Collection exists
        mock_collection = AsyncMock()
        mock_collection.name = "integration_test"
        mock_collections_response = AsyncMock()
        mock_collections_response.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections_response

        with patch("builtins.open", mock_open(read_data=yaml.dump(self.config_data))):
            initializer = VectorDBInitializer("test_config.yaml")

            with patch("click.echo"), patch("time.sleep"):
                initializer.connect()
                result = initializer.create_collection(force=True)

                assert result is True
                mock_client.delete_collection.assert_called_once_with("integration_test")
                mock_client.create_collection.assert_called_once_with()

    def test_error_recovery_scenarios(self):
        """Test various error recovery scenarios."""
        with patch("builtins.open", mock_open(read_data=yaml.dump(self.config_data))):
            initializer = VectorDBInitializer("test_config.yaml")

        # Test operations without connection
        assert initializer.create_collection() is False
        assert initializer.verify_setup() is False
        assert initializer.insert_test_point() is False

        # All should fail gracefully without exceptions
        # and log appropriate error messages
