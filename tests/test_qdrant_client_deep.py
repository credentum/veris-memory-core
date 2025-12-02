#!/usr/bin/env python3
"""
Deep tests for Qdrant client to achieve 40% coverage.

This test suite covers:
- VectorDBInitializer initialization and configuration
- Connection management and SSL configurations
- Collection creation and management
- Setup verification and health checks
- Configuration loading and error handling
- Vector operations and search capabilities
"""

from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from src.storage.qdrant_client import VectorDBInitializer


class TestVectorDBInitializerInitialization:
    """Test VectorDBInitializer initialization and configuration."""

    def test_init_with_config_injection(self):
        """Test initialization with injected configuration."""
        config = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }

        initializer = VectorDBInitializer(config=config, test_mode=True)

        assert initializer.config == config
        assert initializer.test_mode is True
        assert initializer.client is None

    def test_init_without_config_test_mode(self):
        """Test initialization without config in test mode."""
        with patch.object(VectorDBInitializer, "_load_config") as mock_load:
            mock_load.return_value = {"qdrant": {"host": "localhost"}}

            initializer = VectorDBInitializer(config_path="test.yaml", test_mode=True)

            assert initializer.test_mode is True
            mock_load.assert_called_once_with("test.yaml")

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        with patch.object(VectorDBInitializer, "_load_config") as mock_load:
            mock_load.return_value = {}

            initializer = VectorDBInitializer()

            assert initializer.test_mode is False
            assert initializer.client is None
            mock_load.assert_called_once_with(".ctxrc.yaml")


class TestVectorDBInitializerConfigLoading:
    """Test configuration loading methods."""

    def test_load_config_success(self):
        """Test successful configuration loading."""
        config_data = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(config_data))):
            with patch("yaml.safe_load", return_value=config_data):
                initializer = VectorDBInitializer.__new__(VectorDBInitializer)
                initializer.test_mode = False
                result = initializer._load_config("test.yaml")

                assert result == config_data

    def test_load_config_file_not_found_test_mode(self):
        """Test config loading when file not found in test mode."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("storage.qdrant_client.get_test_config") as mock_test_config:
                mock_test_config.return_value = {"test": "config"}

                initializer = VectorDBInitializer.__new__(VectorDBInitializer)
                initializer.test_mode = True
                result = initializer._load_config("nonexistent.yaml")

                assert result == {"test": "config"}
                mock_test_config.assert_called_once()

    def test_load_config_file_not_found_production_mode(self):
        """Test config loading when file not found in production mode."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("click.echo") as mock_echo:
                with patch("sys.exit") as mock_exit:
                    initializer = VectorDBInitializer.__new__(VectorDBInitializer)
                    initializer.test_mode = False

                    initializer._load_config("nonexistent.yaml")

                    mock_echo.assert_called_once()
                    mock_exit.assert_called_once_with(1)

    def test_load_config_yaml_error_production_mode(self):
        """Test config loading with YAML error in production mode."""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content")):
            with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
                with patch("click.echo") as mock_echo:
                    with patch("sys.exit") as mock_exit:
                        initializer = VectorDBInitializer.__new__(VectorDBInitializer)
                        initializer.test_mode = False

                        initializer._load_config("invalid.yaml")

                        mock_echo.assert_called_once()
                        mock_exit.assert_called_once_with(1)

    def test_load_config_non_dict_test_mode(self):
        """Test config loading when result is not a dictionary in test mode."""
        with patch("builtins.open", mock_open(read_data="not_a_dict")):
            with patch("yaml.safe_load", return_value="not_a_dict"):
                with patch("storage.qdrant_client.get_test_config") as mock_test_config:
                    mock_test_config.return_value = {"test": "config"}

                    initializer = VectorDBInitializer.__new__(VectorDBInitializer)
                    initializer.test_mode = True
                    result = initializer._load_config("invalid.yaml")

                    assert result == {"test": "config"}

    def test_load_config_non_dict_production_mode(self):
        """Test config loading when result is not a dictionary in production mode."""
        with patch("builtins.open", mock_open(read_data="not_a_dict")):
            with patch("yaml.safe_load", return_value="not_a_dict"):
                from src.core.config_error import ConfigParseError

                initializer = VectorDBInitializer.__new__(VectorDBInitializer)
                initializer.test_mode = False

                with pytest.raises(ConfigParseError):
                    initializer._load_config("invalid.yaml")


class TestVectorDBInitializerConnection:
    """Test Qdrant connection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }
        self.initializer = VectorDBInitializer(config=config, test_mode=True)

    def test_connect_success_without_ssl(self):
        """Test successful connection to Qdrant without SSL."""
        with patch("storage.qdrant_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "timeout": 5,
            }

            with patch("qdrant_client.QdrantClient") as mock_client_class:
                mock_client = Mock()
                mock_client.get_collections.return_value = Mock()
                mock_client_class.return_value = mock_client

                with patch("click.echo") as mock_echo:
                    result = self.initializer.connect()

                    assert result is True
                    assert self.initializer.client == mock_client
                    mock_client_class.assert_called_once_with(
                        host="localhost", port=6333, timeout=5
                    )
                    mock_client.get_collections.assert_called_once()
                    mock_echo.assert_called_once()

    def test_connect_success_with_ssl(self):
        """Test successful connection to Qdrant with SSL."""
        with patch("storage.qdrant_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "qdrant.example.com",
                "port": 6333,
                "ssl": True,
                "verify_ssl": True,
                "timeout": 10,
            }

            with patch("qdrant_client.QdrantClient") as mock_client_class:
                mock_client = Mock()
                mock_client.get_collections.return_value = Mock()
                mock_client_class.return_value = mock_client

                with patch("click.echo"):
                    result = self.initializer.connect()

                    assert result is True
                    assert self.initializer.client == mock_client
                    mock_client_class.assert_called_once_with(
                        host="qdrant.example.com", port=6333, https=True, verify=True, timeout=10
                    )

    def test_connect_success_with_ssl_no_verify(self):
        """Test successful connection to Qdrant with SSL but no verification."""
        with patch("storage.qdrant_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {
                "host": "qdrant.example.com",
                "port": 6333,
                "ssl": True,
                "verify_ssl": False,
                "timeout": 5,
            }

            with patch("qdrant_client.QdrantClient") as mock_client_class:
                mock_client = Mock()
                mock_client.get_collections.return_value = Mock()
                mock_client_class.return_value = mock_client

                with patch("click.echo"):
                    result = self.initializer.connect()

                    assert result is True
                    mock_client_class.assert_called_once_with(
                        host="qdrant.example.com",
                        port=6333,
                        https=True,
                        verify=False,  # verify_ssl was False
                        timeout=5,
                    )

    def test_connect_failure(self):
        """Test connection failure to Qdrant."""
        with patch("storage.qdrant_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "unreachable.host", "port": 6333, "ssl": False}

            with patch("qdrant_client.QdrantClient", side_effect=Exception("Connection failed")):
                with patch("click.echo") as mock_echo:
                    result = self.initializer.connect()

                    assert result is False
                    assert self.initializer.client is None
                    # Should have called echo with error message
                    error_calls = [call for call in mock_echo.call_args_list if "✗" in str(call)]
                    assert len(error_calls) > 0

    def test_connect_client_none(self):
        """Test connection when client creation returns None."""
        with patch("storage.qdrant_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 6333, "ssl": False}

            with patch("qdrant_client.QdrantClient", return_value=None):
                result = self.initializer.connect()

                assert result is False
                assert self.initializer.client is None


class TestVectorDBInitializerCollectionManagement:
    """Test collection creation and management."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"}
        }
        self.initializer = VectorDBInitializer(config=config, test_mode=True)

    def test_create_collection_success_new(self):
        """Test successful creation of new collection."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []  # No existing collections
        mock_client.get_collections.return_value = mock_collections
        self.initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            with patch("storage.qdrant_client.Config") as mock_config:
                mock_config.EMBEDDING_DIMENSIONS = 384

                result = self.initializer.create_collection()

                assert result is True
                mock_client.create_collection.assert_called_once()
                # Should have called echo for creating collection
                create_calls = [
                    call for call in mock_echo.call_args_list if "Creating" in str(call)
                ]
                assert len(create_calls) > 0

    def test_create_collection_exists_no_force(self):
        """Test collection creation when collection exists without force."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        self.initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_collection(force=False)

            assert result is True
            mock_client.create_collection.assert_not_called()
            # Should have called echo about existing collection
            exists_calls = [
                call for call in mock_echo.call_args_list if "already exists" in str(call)
            ]
            assert len(exists_calls) > 0

    def test_create_collection_exists_with_force(self):
        """Test collection creation when collection exists with force."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        self.initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            with patch("time.sleep") as mock_sleep:
                with patch("storage.qdrant_client.Config") as mock_config:
                    mock_config.EMBEDDING_DIMENSIONS = 384

                    result = self.initializer.create_collection(force=True)

                    assert result is True
                    mock_client.delete_collection.assert_called_once_with("test_collection")
                    mock_client.create_collection.assert_called_once()
                    mock_sleep.assert_called_once_with(1)

    def test_create_collection_no_client(self):
        """Test collection creation when not connected."""
        self.initializer.client = None

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_collection()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    def test_create_collection_exception(self):
        """Test collection creation with exception."""
        mock_client = Mock()
        mock_client.get_collections.side_effect = Exception("Collection error")
        self.initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_collection()

            assert result is False
            # Should have called echo with error message
            error_calls = [call for call in mock_echo.call_args_list if "✗" in str(call)]
            assert len(error_calls) > 0

    def test_create_collection_default_name(self):
        """Test collection creation with default collection name."""
        # Use empty qdrant config to test default collection name
        config = {"qdrant": {}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        initializer.client = mock_client

        with patch("click.echo"):
            with patch("storage.qdrant_client.Config") as mock_config:
                mock_config.EMBEDDING_DIMENSIONS = 384

                result = initializer.create_collection()

                assert result is True
                # Should have used default collection name "project_context"
                mock_client.create_collection.assert_called_once()
                call_args = mock_client.create_collection.call_args
                assert call_args[1]["collection_name"] == "project_context"


class TestVectorDBInitializerVerification:
    """Test setup verification functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {"qdrant": {"collection_name": "test_collection"}}
        self.initializer = VectorDBInitializer(config=config, test_mode=True)

    def test_verify_setup_success_vector_params(self):
        """Test successful setup verification with VectorParams."""
        mock_client = Mock()

        # Mock collection info with VectorParams
        from qdrant_client.models import Distance, VectorParams

        mock_vectors_config = VectorParams(size=384, distance=Distance.COSINE)

        mock_info = Mock()
        mock_info.config = Mock()
        mock_info.config.params = Mock()
        mock_info.config.params.vectors = mock_vectors_config

        mock_client.get_collection.return_value = mock_info
        self.initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is True
            mock_client.get_collection.assert_called_once_with("test_collection")
            # Should have called echo with collection info
            info_calls = [
                call for call in mock_echo.call_args_list if "Collection Info" in str(call)
            ]
            assert len(info_calls) > 0

    def test_verify_setup_success_dict_vectors(self):
        """Test successful setup verification with dict vectors config."""
        mock_client = Mock()

        # Mock collection info with dict vectors config
        mock_vectors_config = {"size": 384, "distance": "Cosine"}

        mock_info = Mock()
        mock_info.config = Mock()
        mock_info.config.params = Mock()
        mock_info.config.params.vectors = mock_vectors_config

        mock_client.get_collection.return_value = mock_info
        self.initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is True
            mock_client.get_collection.assert_called_once_with("test_collection")

    def test_verify_setup_no_client(self):
        """Test setup verification when not connected."""
        self.initializer.client = None

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Qdrant", err=True)

    def test_verify_setup_exception(self):
        """Test setup verification with exception."""
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Verification error")
        self.initializer.client = mock_client

        with patch("click.echo") as mock_echo:
            result = self.initializer.verify_setup()

            assert result is False
            # Should have called echo with error message
            error_calls = [call for call in mock_echo.call_args_list if "✗" in str(call)]
            assert len(error_calls) > 0

    def test_verify_setup_default_collection_name(self):
        """Test setup verification with default collection name."""
        # Use empty qdrant config to test default collection name
        config = {"qdrant": {}}
        initializer = VectorDBInitializer(config=config, test_mode=True)

        mock_client = Mock()
        mock_info = Mock()
        mock_info.config = Mock()
        mock_info.config.params = Mock()
        mock_info.config.params.vectors = Mock()
        mock_client.get_collection.return_value = mock_info
        initializer.client = mock_client

        with patch("click.echo"):
            result = initializer.verify_setup()

            assert result is True
            # Should have used default collection name "project_context"
            mock_client.get_collection.assert_called_once_with("project_context")


class TestVectorDBInitializerIntegration:
    """Test integration scenarios and edge cases."""

    def test_full_initialization_workflow(self):
        """Test complete initialization workflow."""
        config = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "integration_test"}
        }

        initializer = VectorDBInitializer(config=config, test_mode=True)

        # Verify initialization
        assert initializer.config == config
        assert initializer.client is None

        # Test connection (mocked)
        with patch("storage.qdrant_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 6333, "ssl": False}

            with patch("qdrant_client.QdrantClient") as mock_client_class:
                mock_client = Mock()
                mock_client.get_collections.return_value = Mock()
                mock_client_class.return_value = mock_client

                with patch("click.echo"):
                    connection_result = initializer.connect()

                    assert connection_result is True
                    assert initializer.client is not None

    def test_error_handling_chain(self):
        """Test error handling in method chain."""
        initializer = VectorDBInitializer(config={}, test_mode=True)

        # Test collection creation without connection
        with patch("click.echo"):
            collection_result = initializer.create_collection()
            assert collection_result is False

        # Test verification without connection
        with patch("click.echo"):
            verify_result = initializer.verify_setup()
            assert verify_result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
