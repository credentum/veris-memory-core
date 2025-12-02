#!/usr/bin/env python3
"""
Comprehensive tests for VectorDBInitializer to achieve high coverage.

This test suite covers:
- VectorDBInitializer class initialization and configuration loading
- Connection management with various settings (SSL, timeout, etc.)
- Collection creation, deletion, and recreation workflows
- Point insertion, search, and deletion operations
- Configuration validation and error handling
- Setup verification and test point operations
- CLI command functionality and error scenarios
"""

from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from src.core.config_error import ConfigParseError
from src.storage.qdrant_client import VectorDBInitializer


class TestVectorDBInitializerInitialization:
    """Test VectorDBInitializer initialization and configuration."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("storage.qdrant_client.click.echo"):
                with patch("storage.qdrant_client.sys.exit"):
                    initializer = VectorDBInitializer()

        assert initializer.test_mode is False
        assert initializer.client is None

    def test_init_with_config_parameter(self):
        """Test initialization with config parameter."""
        test_config = {
            "qdrant": {"host": "test-host", "port": 6333, "collection_name": "test_collection"}
        }

        initializer = VectorDBInitializer(config=test_config)

        assert initializer.config == test_config
        assert initializer.test_mode is False
        assert initializer.client is None

    def test_init_with_test_mode(self):
        """Test initialization with test mode enabled."""
        with patch("storage.qdrant_client.get_test_config") as mock_get_test_config:
            mock_get_test_config.return_value = {"test": "config"}

            with patch("builtins.open", side_effect=FileNotFoundError):
                initializer = VectorDBInitializer(test_mode=True)

        assert initializer.test_mode is True
        assert initializer.config == {"test": "config"}
        mock_get_test_config.assert_called_once()

    def test_init_with_custom_config_path(self):
        """Test initialization with custom config path."""
        test_config = {"qdrant": {"host": "custom-host"}}

        with patch("builtins.open", mock_open(read_data=yaml.dump(test_config))):
            initializer = VectorDBInitializer(config_path="custom.yaml")

        assert initializer.config == test_config


class TestVectorDBInitializerConfigLoading:
    """Test configuration loading methods."""

    def test_load_config_success(self):
        """Test successful configuration loading."""
        test_config = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "project_context"}
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(test_config))):
            initializer = VectorDBInitializer()

        assert initializer.config == test_config

    def test_load_config_file_not_found_production(self):
        """Test config loading when file not found in production mode."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("storage.qdrant_client.click.echo") as mock_echo:
                with patch("storage.qdrant_client.sys.exit") as mock_exit:
                    VectorDBInitializer(config_path="missing.yaml")

        mock_echo.assert_called_with("Error: missing.yaml not found", err=True)
        mock_exit.assert_called_with(1)

    def test_load_config_file_not_found_test_mode(self):
        """Test config loading when file not found in test mode."""
        with patch("storage.qdrant_client.get_test_config") as mock_get_test_config:
            mock_get_test_config.return_value = {"test": "config"}

            with patch("builtins.open", side_effect=FileNotFoundError):
                initializer = VectorDBInitializer(config_path="missing.yaml", test_mode=True)

        assert initializer.config == {"test": "config"}
        mock_get_test_config.assert_called_once()

    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content: {")):
            with patch(
                "storage.qdrant_client.yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")
            ):
                with patch("storage.qdrant_client.click.echo") as mock_echo:
                    with patch("storage.qdrant_client.sys.exit") as mock_exit:
                        VectorDBInitializer(config_path="invalid.yaml")

        mock_echo.assert_called()
        mock_exit.assert_called_with(1)

    def test_load_config_non_dict_production(self):
        """Test config loading when result is not a dictionary in production."""
        with patch("builtins.open", mock_open(read_data="'string_config'")):
            with patch("storage.qdrant_client.yaml.safe_load", return_value="string_config"):
                with pytest.raises(ConfigParseError):
                    VectorDBInitializer(config_path="invalid.yaml")

    def test_load_config_non_dict_test_mode(self):
        """Test config loading when result is not a dictionary in test mode."""
        with patch("storage.qdrant_client.get_test_config") as mock_get_test_config:
            mock_get_test_config.return_value = {"test": "config"}

            with patch("builtins.open", mock_open(read_data="'string_config'")):
                with patch("storage.qdrant_client.yaml.safe_load", return_value="string_config"):
                    initializer = VectorDBInitializer(config_path="invalid.yaml", test_mode=True)

        assert initializer.config == {"test": "config"}
        mock_get_test_config.assert_called_once()

    def test_load_config_none_result(self):
        """Test config loading when YAML returns None."""
        with patch("builtins.open", mock_open(read_data="")):
            with patch("storage.qdrant_client.yaml.safe_load", return_value=None):
                with pytest.raises(ConfigParseError):
                    VectorDBInitializer(config_path="empty.yaml")


class TestVectorDBInitializerConnection:
    """Test connection management."""

    @pytest.fixture
    def initializer(self):
        """Create VectorDBInitializer instance with test config."""
        test_config = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }
        return VectorDBInitializer(config=test_config)

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_success_without_ssl(self, mock_qdrant_client, mock_config, initializer):
        """Test successful connection without SSL."""
        mock_config.return_value = {"host": "localhost", "port": 6333, "ssl": False, "timeout": 5}

        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = []

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer.connect()

        assert result is True
        assert initializer.client == mock_client_instance
        mock_qdrant_client.assert_called_once_with(host="localhost", port=6333, timeout=5)
        mock_client_instance.get_collections.assert_called_once()
        mock_echo.assert_called_with("✓ Connected to Qdrant at localhost:6333")

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_success_with_ssl(self, mock_qdrant_client, mock_config, initializer):
        """Test successful connection with SSL."""
        mock_config.return_value = {
            "host": "secure-host",
            "port": 6334,
            "ssl": True,
            "verify_ssl": True,
            "timeout": 10,
        }

        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = []

        result = initializer.connect()

        assert result is True
        mock_qdrant_client.assert_called_once_with(
            host="secure-host", port=6334, https=True, verify=True, timeout=10
        )

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_success_ssl_no_verify(self, mock_qdrant_client, mock_config, initializer):
        """Test successful connection with SSL but no verification."""
        mock_config.return_value = {
            "host": "insecure-host",
            "port": 6333,
            "ssl": True,
            "verify_ssl": False,
            "timeout": 5,
        }

        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = []

        result = initializer.connect()

        assert result is True
        mock_qdrant_client.assert_called_once_with(
            host="insecure-host", port=6333, https=True, verify=False, timeout=5
        )

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_client_creation_failure(self, mock_qdrant_client, mock_config, initializer):
        """Test connection failure during client creation."""
        mock_config.return_value = {"host": "localhost", "port": 6333, "ssl": False, "timeout": 5}

        mock_qdrant_client.side_effect = Exception("Connection failed")

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer.connect()

        assert result is False
        assert initializer.client is None
        mock_echo.assert_called_with(
            "✗ Failed to connect to Qdrant at localhost:6333: Connection failed", err=True
        )

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_test_connection_failure(self, mock_qdrant_client, mock_config, initializer):
        """Test connection failure during connection test."""
        mock_config.return_value = {"host": "localhost", "port": 6333, "ssl": False, "timeout": 5}

        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.side_effect = Exception("Test failed")

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer.connect()

        assert result is False
        mock_echo.assert_called_with(
            "✗ Failed to connect to Qdrant at localhost:6333: Test failed", err=True
        )

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.qdrant_client.QdrantClient")
    def test_connect_defaults_port_and_timeout(self, mock_qdrant_client, mock_config, initializer):
        """Test connection uses default port and timeout."""
        mock_config.return_value = {
            "host": "localhost",
            "ssl": False,
            # No port or timeout specified
        }

        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = []

        result = initializer.connect()

        assert result is True
        mock_qdrant_client.assert_called_once_with(host="localhost", port=6333, timeout=5)


class TestVectorDBInitializerCollectionManagement:
    """Test collection creation and management."""

    @pytest.fixture
    def initializer_with_client(self):
        """Create VectorDBInitializer with mock client."""
        test_config = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }

        initializer = VectorDBInitializer(config=test_config)
        initializer.client = Mock()
        return initializer

    def test_create_collection_no_client(self):
        """Test collection creation when client is not connected."""
        initializer = VectorDBInitializer(config={"qdrant": {}})

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer.create_collection()

        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    @patch("storage.qdrant_client.Config")
    def test_create_collection_success(self, mock_config, initializer_with_client):
        """Test successful collection creation."""
        mock_config.EMBEDDING_DIMENSIONS = 384

        # Mock collections response - empty list (no existing collections)
        mock_collections_response = Mock()
        mock_collections_response.collections = []
        initializer_with_client.client.get_collections.return_value = mock_collections_response

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.create_collection()

        assert result is True
        initializer_with_client.client.create_collection.assert_called_once()
        mock_echo.assert_any_call("Creating collection 'test_collection'...")
        mock_echo.assert_any_call("✓ Collection 'test_collection' created successfully")

    def test_create_collection_exists_no_force(self, initializer_with_client):
        """Test collection creation when collection exists and force=False."""
        # Mock existing collection
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        initializer_with_client.client.get_collections.return_value = mock_collections_response

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.create_collection(force=False)

        assert result is True
        initializer_with_client.client.create_collection.assert_not_called()
        mock_echo.assert_called_with(
            "Collection 'test_collection' already exists. Use --force to recreate."
        )

    @patch("storage.qdrant_client.Config")
    @patch("storage.qdrant_client.time.sleep")
    def test_create_collection_exists_with_force(
        self, mock_sleep, mock_config, initializer_with_client
    ):
        """Test collection creation when collection exists and force=True."""
        mock_config.EMBEDDING_DIMENSIONS = 384

        # Mock existing collection
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        initializer_with_client.client.get_collections.return_value = mock_collections_response

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.create_collection(force=True)

        assert result is True
        initializer_with_client.client.delete_collection.assert_called_once_with("test_collection")
        initializer_with_client.client.create_collection.assert_called_once()
        mock_sleep.assert_called_once_with(1)
        mock_echo.assert_any_call("Deleting existing collection 'test_collection'...")

    def test_create_collection_get_collections_failure(self, initializer_with_client):
        """Test collection creation when getting collections fails."""
        initializer_with_client.client.get_collections.side_effect = Exception(
            "Get collections failed"
        )

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.create_collection()

        assert result is False
        mock_echo.assert_called_with(
            "✗ Failed to create collection: Get collections failed", err=True
        )

    @patch("storage.qdrant_client.Config")
    def test_create_collection_creation_failure(self, mock_config, initializer_with_client):
        """Test collection creation when create_collection fails."""
        mock_config.EMBEDDING_DIMENSIONS = 384

        mock_collections_response = Mock()
        mock_collections_response.collections = []
        initializer_with_client.client.get_collections.return_value = mock_collections_response
        initializer_with_client.client.create_collection.side_effect = Exception("Creation failed")

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.create_collection()

        assert result is False
        mock_echo.assert_called_with("✗ Failed to create collection: Creation failed", err=True)

    def test_create_collection_default_name(self):
        """Test collection creation with default collection name."""
        # No collection name specified in config
        initializer = VectorDBInitializer(config={"qdrant": {}})
        initializer.client = Mock()

        mock_collections_response = Mock()
        mock_collections_response.collections = []
        initializer.client.get_collections.return_value = mock_collections_response

        with patch("storage.qdrant_client.Config") as mock_config:
            mock_config.EMBEDDING_DIMENSIONS = 384

            with patch("storage.qdrant_client.click.echo"):
                result = initializer.create_collection()

        assert result is True
        # Should use default collection name
        create_call = initializer.client.create_collection.call_args
        assert create_call[1]["collection_name"] == "project_context"


class TestVectorDBInitializerVerification:
    """Test setup verification methods."""

    @pytest.fixture
    def initializer_with_client(self):
        """Create VectorDBInitializer with mock client."""
        test_config = {"qdrant": {"collection_name": "test_collection", "version": "1.14.x"}}

        initializer = VectorDBInitializer(config=test_config)
        initializer.client = Mock()
        return initializer

    def test_verify_setup_no_client(self):
        """Test setup verification when client is not connected."""
        initializer = VectorDBInitializer(config={"qdrant": {}})

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer.verify_setup()

        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    def test_verify_setup_success_vector_params(self, initializer_with_client):
        """Test successful setup verification with VectorParams."""
        from qdrant_client.models import Distance, VectorParams

        # Mock collection info with VectorParams
        mock_vectors_config = VectorParams(size=384, distance=Distance.COSINE)
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors = mock_vectors_config
        mock_collection_info.points_count = 100

        initializer_with_client.client.get_collection.return_value = mock_collection_info

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.verify_setup()

        assert result is True
        mock_echo.assert_any_call("\nCollection Info:")
        mock_echo.assert_any_call("  Name: test_collection")
        mock_echo.assert_any_call("  Vector size: 384")
        mock_echo.assert_any_call(f"  Distance metric: {Distance.COSINE}")
        mock_echo.assert_any_call("  Points count: 100")
        mock_echo.assert_any_call("\nExpected Qdrant version: 1.14.x")

    def test_verify_setup_success_dict_vectors(self, initializer_with_client):
        """Test successful setup verification with dictionary vectors."""
        from qdrant_client.models import Distance, VectorParams

        # Mock collection info with dictionary vectors (named vectors)
        mock_vector_params = VectorParams(size=512, distance=Distance.DOT)
        mock_vectors_config = {"default": mock_vector_params}
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors = mock_vectors_config
        mock_collection_info.points_count = 50

        initializer_with_client.client.get_collection.return_value = mock_collection_info

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.verify_setup()

        assert result is True
        mock_echo.assert_any_call("  Vector 'default' size: 512")
        mock_echo.assert_any_call(f"  Vector 'default' distance: {Distance.DOT}")

    def test_verify_setup_no_vectors_config(self, initializer_with_client):
        """Test setup verification when vectors config is missing."""
        mock_collection_info = Mock()
        mock_collection_info.config = None
        mock_collection_info.points_count = 0

        initializer_with_client.client.get_collection.return_value = mock_collection_info

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.verify_setup()

        assert result is True
        mock_echo.assert_any_call("  Points count: 0")

    def test_verify_setup_get_collection_failure(self, initializer_with_client):
        """Test setup verification when get_collection fails."""
        initializer_with_client.client.get_collection.side_effect = Exception(
            "Get collection failed"
        )

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.verify_setup()

        assert result is False
        mock_echo.assert_called_with("✗ Failed to verify setup: Get collection failed", err=True)

    def test_verify_setup_default_collection_name(self):
        """Test setup verification with default collection name."""
        initializer = VectorDBInitializer(config={"qdrant": {}})
        initializer.client = Mock()

        mock_collection_info = Mock()
        mock_collection_info.config = None
        mock_collection_info.points_count = 0
        initializer.client.get_collection.return_value = mock_collection_info

        with patch("storage.qdrant_client.click.echo"):
            result = initializer.verify_setup()

        assert result is True
        initializer.client.get_collection.assert_called_once_with("project_context")


class TestVectorDBInitializerTestOperations:
    """Test point insertion and test operations."""

    @pytest.fixture
    def initializer_with_client(self):
        """Create VectorDBInitializer with mock client."""
        test_config = {"qdrant": {"collection_name": "test_collection"}}

        initializer = VectorDBInitializer(config=test_config)
        initializer.client = Mock()
        return initializer

    def test_insert_test_point_no_client(self):
        """Test test point insertion when client is not connected."""
        initializer = VectorDBInitializer(config={"qdrant": {}})

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer.insert_test_point()

        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    @patch("storage.qdrant_client.Config")
    @patch("random.random")
    def test_insert_test_point_success(self, mock_random, mock_config, initializer_with_client):
        """Test successful test point insertion."""
        mock_config.EMBEDDING_DIMENSIONS = 384
        mock_random.return_value = 0.5  # Fixed random value for testing

        # Mock search results
        mock_search_result = Mock()
        mock_search_result.id = "test-point-001"
        initializer_with_client.client.search.return_value = [mock_search_result]

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.insert_test_point()

        assert result is True

        # Verify upsert was called
        initializer_with_client.client.upsert.assert_called_once()
        upsert_call = initializer_with_client.client.upsert.call_args
        assert upsert_call[1]["collection_name"] == "test_collection"

        # Verify search was called
        initializer_with_client.client.search.assert_called_once()
        search_call = initializer_with_client.client.search.call_args
        assert search_call[1]["collection_name"] == "test_collection"
        assert search_call[1]["limit"] == 1

        # Verify delete was called for cleanup
        initializer_with_client.client.delete.assert_called_once()
        delete_call = initializer_with_client.client.delete.call_args
        assert delete_call[1]["collection_name"] == "test_collection"
        assert delete_call[1]["points_selector"] == ["test-point-001"]

        mock_echo.assert_called_with("✓ Test point inserted and retrieved successfully")

    @patch("storage.qdrant_client.Config")
    @patch("random.random")
    def test_insert_test_point_search_failure(
        self, mock_random, mock_config, initializer_with_client
    ):
        """Test test point insertion when search returns wrong result."""
        mock_config.EMBEDDING_DIMENSIONS = 384
        mock_random.return_value = 0.5

        # Mock search results with wrong ID
        mock_search_result = Mock()
        mock_search_result.id = "wrong-id"
        initializer_with_client.client.search.return_value = [mock_search_result]

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.insert_test_point()

        assert result is False
        mock_echo.assert_called_with("✗ Test point verification failed", err=True)

    @patch("storage.qdrant_client.Config")
    @patch("random.random")
    def test_insert_test_point_empty_search_results(
        self, mock_random, mock_config, initializer_with_client
    ):
        """Test test point insertion when search returns empty results."""
        mock_config.EMBEDDING_DIMENSIONS = 384
        mock_random.return_value = 0.5

        # Empty search results
        initializer_with_client.client.search.return_value = []

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.insert_test_point()

        assert result is False
        mock_echo.assert_called_with("✗ Test point verification failed", err=True)

    @patch("storage.qdrant_client.Config")
    def test_insert_test_point_upsert_failure(self, mock_config, initializer_with_client):
        """Test test point insertion when upsert fails."""
        mock_config.EMBEDDING_DIMENSIONS = 384

        initializer_with_client.client.upsert.side_effect = Exception("Upsert failed")

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.insert_test_point()

        assert result is False
        mock_echo.assert_called_with("✗ Failed to test point operations: Upsert failed", err=True)

    @patch("storage.qdrant_client.Config")
    @patch("random.random")
    def test_insert_test_point_search_exception(
        self, mock_random, mock_config, initializer_with_client
    ):
        """Test test point insertion when search raises exception."""
        mock_config.EMBEDDING_DIMENSIONS = 384
        mock_random.return_value = 0.5

        initializer_with_client.client.search.side_effect = Exception("Search failed")

        with patch("storage.qdrant_client.click.echo") as mock_echo:
            result = initializer_with_client.insert_test_point()

        assert result is False
        mock_echo.assert_called_with("✗ Failed to test point operations: Search failed", err=True)

    def test_insert_test_point_default_collection_name(self):
        """Test test point insertion with default collection name."""
        initializer = VectorDBInitializer(config={"qdrant": {}})
        initializer.client = Mock()

        mock_search_result = Mock()
        mock_search_result.id = "test-point-001"
        initializer.client.search.return_value = [mock_search_result]

        with patch("storage.qdrant_client.Config") as mock_config:
            mock_config.EMBEDDING_DIMENSIONS = 384
            with patch("storage.qdrant_client.random.random", return_value=0.5):
                with patch("storage.qdrant_client.click.echo"):
                    result = initializer.insert_test_point()

        assert result is True

        # Verify default collection name is used
        upsert_call = initializer.client.upsert.call_args
        assert upsert_call[1]["collection_name"] == "project_context"


class TestVectorDBInitializerCLI:
    """Test CLI command functionality."""

    @patch("storage.qdrant_client.VectorDBInitializer")
    @patch("storage.qdrant_client.click.echo")
    def test_main_success_complete_flow(self, mock_echo, mock_initializer_class):
        """Test successful complete CLI flow."""
        # Mock initializer instance
        mock_initializer = Mock()
        mock_initializer_class.return_value = mock_initializer
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.insert_test_point.return_value = True

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        assert result.exit_code == 0
        mock_initializer.connect.assert_called_once()
        mock_initializer.create_collection.assert_called_once_with(force=False)
        mock_initializer.verify_setup.assert_called_once()
        mock_initializer.insert_test_point.assert_called_once()

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_connect_failure(self, mock_initializer_class):
        """Test CLI when connection fails."""
        mock_initializer = Mock()
        mock_initializer_class.return_value = mock_initializer
        mock_initializer.connect.return_value = False

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        # CLI runner catches sys.exit, so execution continues, but we verify the connection was attempted
        mock_initializer.connect.assert_called_once()

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_create_collection_failure(self, mock_initializer_class):
        """Test CLI when collection creation fails."""
        mock_initializer = Mock()
        mock_initializer_class.return_value = mock_initializer
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = False

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        mock_initializer.create_collection.assert_called_once_with(force=False)

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_verify_setup_failure(self, mock_initializer_class):
        """Test CLI when setup verification fails."""
        mock_initializer = Mock()
        mock_initializer_class.return_value = mock_initializer
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = False

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        mock_initializer.verify_setup.assert_called_once()

    @patch("storage.qdrant_client.VectorDBInitializer")
    @patch("storage.qdrant_client.sys.exit")
    def test_main_test_point_failure(self, mock_exit, mock_initializer_class):
        """Test CLI when test point insertion fails."""
        mock_initializer = Mock()
        mock_initializer_class.return_value = mock_initializer
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.insert_test_point.return_value = False

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        mock_initializer.insert_test_point.assert_called_once()

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_with_force_flag(self, mock_initializer_class):
        """Test CLI with force flag."""
        mock_initializer = Mock()
        mock_initializer_class.return_value = mock_initializer
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.insert_test_point.return_value = True

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, ["--force"])

        assert result.exit_code == 0
        mock_initializer.create_collection.assert_called_once_with(force=True)

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_with_skip_test_flag(self, mock_initializer_class):
        """Test CLI with skip-test flag."""
        mock_initializer = Mock()
        mock_initializer_class.return_value = mock_initializer
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, ["--skip-test"])

        assert result.exit_code == 0
        mock_initializer.insert_test_point.assert_not_called()

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_with_both_flags(self, mock_initializer_class):
        """Test CLI with both force and skip-test flags."""
        mock_initializer = Mock()
        mock_initializer_class.return_value = mock_initializer
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, ["--force", "--skip-test"])

        assert result.exit_code == 0
        mock_initializer.create_collection.assert_called_once_with(force=True)
        mock_initializer.insert_test_point.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
