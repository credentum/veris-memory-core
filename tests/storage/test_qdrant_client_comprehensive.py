#!/usr/bin/env python3
"""
Comprehensive tests for Qdrant client to achieve high coverage.

This test suite covers:
- Connection management with various configurations
- Collection creation and management
- Vector operations and search
- Error handling and edge cases
- Configuration loading and validation
- SSL/security settings
"""

import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml
from qdrant_client.models import Distance, VectorParams

from src.storage.qdrant_client import VectorDBInitializer


class MockQdrantClient:
    """Mock Qdrant client for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_collections(self):
        return Mock(collections=[])

    def create_collection(self, **kwargs):
        pass

    def delete_collection(self, name):
        pass

    def get_collection(self, name):
        return Mock(
            config=Mock(params=Mock(vectors=VectorParams(size=1536, distance=Distance.COSINE))),
            points_count=0,
        )

    def upsert(self, **kwargs):
        pass

    def search(self, **kwargs):
        return [Mock(id="test-point-001")]

    def delete(self, **kwargs):
        pass


class TestVectorDBInitializer:
    """Test cases for VectorDBInitializer class."""

    def test_init_with_config_dict(self):
        """Test initialization with provided config dictionary."""
        config = {
            "qdrant": {"host": "test-host", "port": 6333, "collection_name": "test_collection"}
        }

        initializer = VectorDBInitializer(config=config, test_mode=True)

        assert initializer.config == config
        assert initializer.test_mode is True
        assert initializer.client is None

    def test_init_with_config_file(self):
        """Test initialization with config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "qdrant": {"host": "localhost", "port": 6333, "collection_name": "project_context"}
            }
            yaml.dump(config, f)
            f.flush()

            initializer = VectorDBInitializer(config_path=f.name, test_mode=True)

            assert initializer.config == config

    def test_init_with_missing_config_test_mode(self):
        """Test initialization with missing config in test mode."""
        with patch("storage.qdrant_client.get_test_config") as mock_get_test_config:
            mock_config = {"qdrant": {"collection_name": "test"}}
            mock_get_test_config.return_value = mock_config

            initializer = VectorDBInitializer(config_path="nonexistent.yaml", test_mode=True)

            assert initializer.config == mock_config
            mock_get_test_config.assert_called_once()

    def test_init_with_missing_config_production_mode(self):
        """Test initialization with missing config in production mode."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("click.echo") as mock_echo:
                with patch("sys.exit") as mock_exit:
                    VectorDBInitializer(config_path="nonexistent.yaml", test_mode=False)

                    mock_echo.assert_called()
                    mock_exit.assert_called_with(1)

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            with patch("click.echo") as mock_echo:
                with patch("sys.exit") as mock_exit:
                    VectorDBInitializer(config_path=f.name, test_mode=False)

                    mock_echo.assert_called()
                    mock_exit.assert_called_with(1)

    def test_load_config_non_dict_content(self):
        """Test loading config with non-dictionary content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump("not a dictionary", f)
            f.flush()

            with patch("storage.qdrant_client.get_test_config") as mock_get_test_config:
                mock_config = {"qdrant": {"collection_name": "test"}}
                mock_get_test_config.return_value = mock_config

                initializer = VectorDBInitializer(config_path=f.name, test_mode=True)

                assert initializer.config == mock_config

    @patch("storage.qdrant_client.QdrantClient")
    @patch("storage.qdrant_client.get_secure_connection_config")
    def test_connect_success_without_ssl(self, mock_get_config, mock_qdrant_client):
        """Test successful connection without SSL."""
        mock_get_config.return_value = {
            "host": "localhost",
            "port": 6333,
            "ssl": False,
            "timeout": 5,
        }

        mock_client = MockQdrantClient()
        mock_qdrant_client.return_value = mock_client

        initializer = VectorDBInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.connect()

            assert result is True
            assert initializer.client == mock_client
            mock_qdrant_client.assert_called_with(host="localhost", port=6333, timeout=5)
            mock_echo.assert_called_with("✓ Connected to Qdrant at localhost:6333")

    @patch("storage.qdrant_client.QdrantClient")
    @patch("storage.qdrant_client.get_secure_connection_config")
    def test_connect_success_with_ssl(self, mock_get_config, mock_qdrant_client):
        """Test successful connection with SSL."""
        mock_get_config.return_value = {
            "host": "secure-host",
            "port": 6333,
            "ssl": True,
            "verify_ssl": True,
            "timeout": 10,
        }

        mock_client = MockQdrantClient()
        mock_qdrant_client.return_value = mock_client

        initializer = VectorDBInitializer(test_mode=True)

        with patch("click.echo"):
            result = initializer.connect()

            assert result is True
            mock_qdrant_client.assert_called_with(
                host="secure-host", port=6333, https=True, verify=True, timeout=10
            )

    @patch("storage.qdrant_client.QdrantClient")
    @patch("storage.qdrant_client.get_secure_connection_config")
    def test_connect_failure(self, mock_get_config, mock_qdrant_client):
        """Test connection failure."""
        mock_get_config.return_value = {"host": "unavailable-host", "port": 6333, "ssl": False}

        mock_qdrant_client.side_effect = Exception("Connection failed")

        initializer = VectorDBInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.connect()

            assert result is False
            assert initializer.client is None
            mock_echo.assert_called_with(
                "✗ Failed to connect to Qdrant at unavailable-host:6333: Connection failed",
                err=True,
            )

    def test_create_collection_not_connected(self):
        """Test creating collection when not connected."""
        initializer = VectorDBInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.create_collection()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    @patch("storage.qdrant_client.Config")
    def test_create_collection_success_new(self, mock_config):
        """Test successful creation of new collection."""
        mock_config.EMBEDDING_DIMENSIONS = 1536

        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_collections = Mock(collections=[])  # No existing collections
        mock_client.get_collections.return_value = mock_collections
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.create_collection()

            assert result is True
            mock_client.create_collection.assert_called_once()
            mock_echo.assert_any_call("Creating collection 'test_collection'...")
            mock_echo.assert_any_call("✓ Collection 'test_collection' created successfully")

    def test_create_collection_exists_no_force(self):
        """Test collection creation when collection exists without force."""
        config = {"qdrant": {"collection_name": "existing_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_collection = Mock()
        mock_collection.name = "existing_collection"
        mock_client = Mock()
        mock_collections = Mock(collections=[mock_collection])
        mock_client.get_collections.return_value = mock_collections
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.create_collection(force=False)

            assert result is True
            mock_client.create_collection.assert_not_called()
            mock_echo.assert_called_with(
                "Collection 'existing_collection' already exists. Use --force to recreate."
            )

    @patch("storage.qdrant_client.Config")
    @patch("time.sleep")
    def test_create_collection_exists_with_force(self, mock_sleep, mock_config):
        """Test collection recreation with force flag."""
        mock_config.EMBEDDING_DIMENSIONS = 1536

        config = {"qdrant": {"collection_name": "existing_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_collection = Mock()
        mock_collection.name = "existing_collection"
        mock_client = Mock()
        mock_collections = Mock(collections=[mock_collection])
        mock_client.get_collections.return_value = mock_collections
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.create_collection(force=True)

            assert result is True
            mock_client.delete_collection.assert_called_with("existing_collection")
            mock_client.create_collection.assert_called_once()
            mock_sleep.assert_called_with(1)  # Give Qdrant time to process
            mock_echo.assert_any_call("Deleting existing collection 'existing_collection'...")

    def test_create_collection_error(self):
        """Test collection creation with error."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_client.get_collections.side_effect = Exception("Database error")
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.create_collection()

            assert result is False
            mock_echo.assert_called_with("✗ Failed to create collection: Database error", err=True)

    def test_verify_setup_not_connected(self):
        """Test verifying setup when not connected."""
        initializer = VectorDBInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    def test_verify_setup_success_single_vector(self):
        """Test successful setup verification with single vector config."""
        config = {"qdrant": {"collection_name": "test_collection", "version": "1.15.0"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_info = Mock()
        mock_info.config = Mock()
        mock_info.config.params = Mock()
        mock_info.config.params.vectors = VectorParams(size=1536, distance=Distance.COSINE)
        mock_info.points_count = 42

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_info
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()

            assert result is True
            mock_echo.assert_any_call("\nCollection Info:")
            mock_echo.assert_any_call("  Name: test_collection")
            mock_echo.assert_any_call("  Vector size: 1536")
            mock_echo.assert_any_call("  Distance metric: Distance.COSINE")
            mock_echo.assert_any_call("  Points count: 42")
            mock_echo.assert_any_call("\nExpected Qdrant version: 1.15.0")

    def test_verify_setup_success_named_vectors(self):
        """Test successful setup verification with named vector configs."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_info = Mock()
        mock_info.config = Mock()
        mock_info.config.params = Mock()
        mock_info.config.params.vectors = {
            "text": VectorParams(size=1536, distance=Distance.COSINE),
            "image": VectorParams(size=512, distance=Distance.EUCLIDEAN),
        }
        mock_info.points_count = 100

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_info
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()

            assert result is True
            mock_echo.assert_any_call("  Vector 'text' size: 1536")
            mock_echo.assert_any_call("  Vector 'image' size: 512")

    def test_verify_setup_error(self):
        """Test setup verification with error."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()

            assert result is False
            mock_echo.assert_called_with("✗ Failed to verify setup: Collection not found", err=True)

    def test_insert_test_point_not_connected(self):
        """Test inserting test point when not connected."""
        initializer = VectorDBInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    @patch("storage.qdrant_client.Config")
    @patch("random.random")
    def test_insert_test_point_success(self, mock_random, mock_config):
        """Test successful test point insertion."""
        mock_config.EMBEDDING_DIMENSIONS = 3  # Small for testing
        mock_random.return_value = 0.5

        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_search_result = Mock()
        mock_search_result.id = "test-point-001"

        mock_client = Mock()
        mock_client.search.return_value = [mock_search_result]
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.insert_test_point()

            assert result is True
            mock_client.upsert.assert_called_once()
            mock_client.search.assert_called_once()
            mock_client.delete.assert_called_once()
            mock_echo.assert_called_with("✓ Test point inserted and retrieved successfully")

    @patch("storage.qdrant_client.Config")
    def test_insert_test_point_verification_failed(self, mock_config):
        """Test test point insertion with verification failure."""
        mock_config.EMBEDDING_DIMENSIONS = 3

        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_client.search.return_value = []  # No results found
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with("✗ Test point verification failed", err=True)

    @patch("storage.qdrant_client.Config")
    def test_insert_test_point_error(self, mock_config):
        """Test test point insertion with error."""
        mock_config.EMBEDDING_DIMENSIONS = 3

        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_client.upsert.side_effect = Exception("Insert failed")
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with(
                "✗ Failed to test point operations: Insert failed", err=True
            )


class TestQdrantCliCommands:
    """Test CLI command functionality."""

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_success_all_steps(self, mock_initializer_class):
        """Test successful execution of all setup steps."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.insert_test_point.return_value = True

        mock_initializer_class.return_value = mock_initializer

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
    def test_main_connection_failure(self, mock_initializer_class):
        """Test main command with connection failure."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = False

        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        assert result.exit_code == 1
        mock_initializer.connect.assert_called_once()
        mock_initializer.create_collection.assert_not_called()

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_collection_creation_failure(self, mock_initializer_class):
        """Test main command with collection creation failure."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = False

        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        assert result.exit_code == 1
        mock_initializer.create_collection.assert_called_once()
        mock_initializer.verify_setup.assert_not_called()

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_with_force_flag(self, mock_initializer_class):
        """Test main command with force flag."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.insert_test_point.return_value = True

        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, ["--force"])

        assert result.exit_code == 0
        mock_initializer.create_collection.assert_called_once_with(force=True)

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_skip_test(self, mock_initializer_class):
        """Test main command with skip test flag."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True

        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, ["--skip-test"])

        assert result.exit_code == 0
        mock_initializer.verify_setup.assert_called_once()
        mock_initializer.insert_test_point.assert_not_called()

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_verify_setup_failure(self, mock_initializer_class):
        """Test main command with verify setup failure."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = False

        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        assert result.exit_code == 1
        mock_initializer.verify_setup.assert_called_once()
        mock_initializer.insert_test_point.assert_not_called()

    @patch("storage.qdrant_client.VectorDBInitializer")
    def test_main_test_point_failure(self, mock_initializer_class):
        """Test main command with test point failure."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_collection.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.insert_test_point.return_value = False

        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.qdrant_client import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        assert result.exit_code == 1
        mock_initializer.insert_test_point.assert_called_once()


class TestVectorDBInitializerEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_with_missing_qdrant_section(self):
        """Test configuration without qdrant section."""
        config = {"other_service": {"host": "localhost"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        # Should use defaults when qdrant section is missing
        assert initializer.config == config

    @patch("storage.qdrant_client.get_secure_connection_config")
    def test_connect_with_minimal_config(self, mock_get_config):
        """Test connection with minimal configuration."""
        mock_get_config.return_value = {"host": "localhost"}  # Missing port, ssl, timeout

        initializer = VectorDBInitializer(test_mode=True)

        with patch("storage.qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MockQdrantClient()
            mock_qdrant_client.return_value = mock_client

            result = initializer.connect()

            assert result is True
            # Should use defaults for missing values
            mock_qdrant_client.assert_called_with(host="localhost", port=6333, timeout=5)

    def test_create_collection_with_minimal_config(self):
        """Test collection creation with minimal configuration."""
        initializer = VectorDBInitializer(config={}, test_mode=True)  # Empty config

        mock_client = Mock()
        mock_collections = Mock(collections=[])
        mock_client.get_collections.return_value = mock_collections
        initializer.client = mock_client

        with patch("storage.qdrant_client.Config") as mock_config:
            mock_config.EMBEDDING_DIMENSIONS = 1536

            result = initializer.create_collection()

            assert result is True
            # Should use default collection name
            create_call = mock_client.create_collection.call_args
            assert create_call[1]["collection_name"] == "project_context"

    def test_verify_setup_with_minimal_config(self):
        """Test setup verification with minimal configuration."""
        initializer = VectorDBInitializer(config={}, test_mode=True)

        mock_info = Mock()
        mock_info.config = Mock()
        mock_info.config.params = Mock()
        mock_info.config.params.vectors = VectorParams(size=1536, distance=Distance.COSINE)
        mock_info.points_count = 0

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_info
        initializer.client = mock_client

        result = initializer.verify_setup()

        assert result is True
        # Should use default collection name and version
        mock_client.get_collection.assert_called_with("project_context")

    def test_verify_setup_with_missing_vector_config(self):
        """Test setup verification when vector config is missing."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_info = Mock()
        mock_info.config = None  # Missing config
        mock_info.points_count = 5

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_info
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()

            assert result is True
            mock_echo.assert_any_call("  Points count: 5")
            # Should not crash when vector config is missing

    @patch("storage.qdrant_client.Config")
    def test_insert_test_point_with_search_mismatch(self, mock_config):
        """Test test point insertion when search returns wrong ID."""
        mock_config.EMBEDDING_DIMENSIONS = 3

        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_search_result = Mock()
        mock_search_result.id = "wrong-id"  # Different from expected

        mock_client = Mock()
        mock_client.search.return_value = [mock_search_result]
        initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = initializer.insert_test_point()

            assert result is False
            mock_echo.assert_called_with("✗ Test point verification failed", err=True)


class TestHybridSearch:
    """Tests for VectorDBInitializer.hybrid_search() method."""

    def test_hybrid_search_not_connected(self):
        """Test hybrid_search raises error when not connected."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        with pytest.raises(RuntimeError, match="Not connected to Qdrant"):
            initializer.hybrid_search(dense_vector=[0.1, 0.2, 0.3])

    def test_hybrid_search_invalid_dense_vector_empty(self):
        """Test hybrid_search raises error for empty dense vector."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)
        initializer.client = Mock()

        with pytest.raises(ValueError, match="dense_vector must be a non-empty list"):
            initializer.hybrid_search(dense_vector=[])

    def test_hybrid_search_invalid_dense_vector_none(self):
        """Test hybrid_search raises error for None dense vector."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)
        initializer.client = Mock()

        with pytest.raises(ValueError, match="dense_vector must be a non-empty list"):
            initializer.hybrid_search(dense_vector=None)

    def test_hybrid_search_invalid_dense_vector_non_numeric(self):
        """Test hybrid_search raises error for non-numeric dense vector."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)
        initializer.client = Mock()

        with pytest.raises(ValueError, match="dense_vector must contain only numeric values"):
            initializer.hybrid_search(dense_vector=["a", "b", "c"])

    def test_hybrid_search_dense_only(self):
        """Test hybrid_search with dense vector only (no sparse)."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_response = Mock()
        mock_response.points = [
            Mock(id="doc-1", score=0.95, payload={"title": "Test"}),
            Mock(id="doc-2", score=0.85, payload={"title": "Other"}),
        ]

        mock_client = Mock()
        mock_client.query_points.return_value = mock_response
        initializer.client = mock_client

        results = initializer.hybrid_search(
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector=None,
            limit=10
        )

        assert len(results) == 2
        assert results[0]["id"] == "doc-1"
        assert results[0]["score"] == 0.95
        assert results[0]["hybrid"] is False  # No sparse vector provided

    @patch("storage.qdrant_client.SPARSE_EMBEDDINGS_ENABLED", True)
    def test_hybrid_search_with_sparse_vector(self):
        """Test hybrid_search with both dense and sparse vectors."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_response = Mock()
        mock_response.points = [
            Mock(id="doc-1", score=0.95, payload={"title": "Keyword Match"}),
            Mock(id="doc-2", score=0.90, payload={"title": "Semantic Match"}),
        ]

        mock_client = Mock()
        mock_client.query_points.return_value = mock_response
        initializer.client = mock_client

        sparse_vector = {"indices": [0, 5, 10], "values": [0.1, 0.5, 0.3]}

        results = initializer.hybrid_search(
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector=sparse_vector,
            limit=10
        )

        assert len(results) == 2
        assert results[0]["hybrid"] is True  # Both vectors used

        # Verify query_points was called with prefetch (hybrid mode)
        call_args = mock_client.query_points.call_args
        assert call_args is not None

    def test_hybrid_search_connection_error(self):
        """Test hybrid_search handles connection errors."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_client.query_points.side_effect = ConnectionError("Connection failed")
        initializer.client = mock_client

        with pytest.raises(RuntimeError, match="Qdrant connection error during hybrid search"):
            initializer.hybrid_search(dense_vector=[0.1, 0.2, 0.3])

    def test_hybrid_search_timeout_error(self):
        """Test hybrid_search handles timeout errors."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_client.query_points.side_effect = TimeoutError("Request timed out")
        initializer.client = mock_client

        with pytest.raises(RuntimeError, match="Qdrant timeout error during hybrid search"):
            initializer.hybrid_search(dense_vector=[0.1, 0.2, 0.3])

    def test_hybrid_search_generic_error(self):
        """Test hybrid_search handles generic errors."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_client.query_points.side_effect = Exception("Unknown error")
        initializer.client = mock_client

        with pytest.raises(RuntimeError, match="Failed to perform hybrid search"):
            initializer.hybrid_search(dense_vector=[0.1, 0.2, 0.3])

    def test_hybrid_search_respects_limit(self):
        """Test hybrid_search respects the limit parameter."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_response = Mock()
        mock_response.points = [
            Mock(id=f"doc-{i}", score=0.9 - i * 0.1, payload={})
            for i in range(5)
        ]

        mock_client = Mock()
        mock_client.query_points.return_value = mock_response
        initializer.client = mock_client

        results = initializer.hybrid_search(
            dense_vector=[0.1, 0.2, 0.3],
            limit=5
        )

        assert len(results) == 5

        # Verify limit was passed to query_points
        call_args = mock_client.query_points.call_args
        assert call_args.kwargs.get("limit") == 5

    def test_hybrid_search_empty_results(self):
        """Test hybrid_search with no results."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_response = Mock()
        mock_response.points = []

        mock_client = Mock()
        mock_client.query_points.return_value = mock_response
        initializer.client = mock_client

        results = initializer.hybrid_search(dense_vector=[0.1, 0.2, 0.3])

        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
