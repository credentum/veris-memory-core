#!/usr/bin/env python3
"""
Comprehensive tests for the Qdrant client to increase coverage.

This test suite covers vector database initialization and operations
to achieve high code coverage for the storage.qdrant_client module.
"""

import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.storage.qdrant_client import VectorDBInitializer  # noqa: E402


class TestVectorDBInitializer:
    """Test suite for VectorDBInitializer class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {
                "qdrant": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "test_collection",
                    "dimensions": 1536,
                }
            }
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = VectorDBInitializer(config_path=config_file.name)
                assert initializer.config == config
                assert initializer.client is None
            finally:
                os.unlink(config_file.name)

    def test_init_missing_config_file(self):
        """Test initialization with missing configuration file."""
        with pytest.raises(SystemExit):
            VectorDBInitializer(config_path="/nonexistent/config.yaml")

    def test_init_invalid_yaml(self):
        """Test initialization with invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config_file.write("invalid: yaml: content: [")
            config_file.flush()

            try:
                with pytest.raises(SystemExit):
                    VectorDBInitializer(config_path=config_file.name)
            finally:
                os.unlink(config_file.name)

    def test_init_non_dict_config(self):
        """Test initialization with non-dictionary configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            yaml.dump(["not", "a", "dict"], config_file)
            config_file.flush()

            try:
                with pytest.raises(SystemExit):
                    VectorDBInitializer(config_path=config_file.name)
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_connect_success_default_url(self, mock_qdrant_client):
        """Test successful connection with default URL."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"host": "localhost", "port": 6333}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = await initializer.connect()

                assert result is True
                assert initializer.client == mock_client
                mock_qdrant_client.assert_called_with(url="http://localhost:6333")
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_connect_success_with_url_from_config(self, mock_qdrant_client):
        """Test successful connection with URL from configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"url": "http://custom:6333"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = await initializer.connect()

                assert result is True
                mock_qdrant_client.assert_called_with(url="http://custom:6333")
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_connect_success_with_url_from_env(self, mock_qdrant_client):
        """Test successful connection with URL from environment variable."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_qdrant_client.return_value = mock_client

                with patch.dict(os.environ, {"QDRANT_URL": "http://env:6333"}):
                    initializer = VectorDBInitializer(config_path=config_file.name)
                    result = await initializer.connect()

                assert result is True
                mock_qdrant_client.assert_called_with(url="http://env:6333")
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_connect_with_https(self, mock_qdrant_client):
        """Test connection with HTTPS enabled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"host": "localhost", "port": 6333}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = initializer.connect(https=True)

                assert result is True
                mock_qdrant_client.assert_called_with(url="https://localhost:6333")
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_connect_with_api_key(self, mock_qdrant_client):
        """Test connection with API key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"host": "localhost", "port": 6333}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = initializer.connect(api_key="test_api_key")

                assert result is True
                mock_qdrant_client.assert_called_with(
                    url="http://localhost:6333", api_key="test_api_key"
                )
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_connect_with_verify_false(self, mock_qdrant_client):
        """Test connection with SSL verification disabled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"host": "localhost", "port": 6333}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = initializer.connect(verify=False)

                assert result is True
                mock_qdrant_client.assert_called_with(url="http://localhost:6333", verify=False)
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_connect_with_verify_path(self, mock_qdrant_client):
        """Test connection with custom verification path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"host": "localhost", "port": 6333}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = initializer.connect(verify="/path/to/ca.crt")

                assert result is True
                mock_qdrant_client.assert_called_with(
                    url="http://localhost:6333", verify="/path/to/ca.crt"
                )
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_connect_failure(self, mock_qdrant_client):
        """Test connection failure handling."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"host": "localhost", "port": 6333}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_qdrant_client.side_effect = Exception("Connection failed")

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = await initializer.connect()

                assert result is False
                assert initializer.client is None
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_create_collection_success(self, mock_qdrant_client):
        """Test successful collection creation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {
                "qdrant": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "test_collection",
                    "dimensions": 768,
                }
            }
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_client.get_collection.side_effect = Exception("Collection not found")
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                await initializer.connect()
                result = initializer.create_collection()

                assert result is True
                mock_client.create_collection.assert_called_once_with()
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_create_collection_already_exists(self, mock_qdrant_client):
        """Test collection creation when collection already exists."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {
                "qdrant": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "test_collection",
                    "dimensions": 768,
                }
            }
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_client.get_collection.return_value = {"name": "test_collection"}
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                await initializer.connect()
                result = initializer.create_collection()

                assert result is True
                mock_client.create_collection.assert_not_called()
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_create_collection_no_client(self, mock_qdrant_client):
        """Test collection creation without connected client."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"collection_name": "test_collection"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = VectorDBInitializer(config_path=config_file.name)
                result = initializer.create_collection()

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_create_collection_creation_failure(self, mock_qdrant_client):
        """Test collection creation failure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {
                "qdrant": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "test_collection",
                    "dimensions": 768,
                }
            }
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_client.get_collection.side_effect = Exception("Collection not found")
                mock_client.create_collection.side_effect = Exception("Creation failed")
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                await initializer.connect()
                result = initializer.create_collection()

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_initialize_success(self, mock_qdrant_client):
        """Test successful initialization (connect + create collection)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {
                "qdrant": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "test_collection",
                    "dimensions": 768,
                }
            }
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_client.get_collection.side_effect = Exception("Collection not found")
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = await initializer.initialize()

                assert result is True
                assert initializer.client is not None
                mock_client.create_collection.assert_called_once_with()
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_initialize_connect_failure(self, mock_qdrant_client):
        """Test initialization failure when connection fails."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"host": "localhost", "port": 6333}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_qdrant_client.side_effect = Exception("Connection failed")

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = await initializer.initialize()

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_initialize_collection_failure(self, mock_qdrant_client):
        """Test initialization failure when collection creation fails."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {
                "qdrant": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "test_collection",
                    "dimensions": 768,
                }
            }
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_client.get_collection.side_effect = Exception("Collection not found")
                mock_client.create_collection.side_effect = Exception("Creation failed")
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = await initializer.initialize()

                assert result is False
            finally:
                os.unlink(config_file.name)


class TestVectorDBInitializerConfigHandling:
    """Test suite for configuration handling edge cases."""

    def test_missing_qdrant_section(self):
        """Test initialization with missing qdrant configuration section."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"other_section": {"key": "value"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = VectorDBInitializer(config_path=config_file.name)
                # Should handle missing qdrant section gracefully
                assert "qdrant" not in initializer.config
            finally:
                os.unlink(config_file.name)

    def test_empty_qdrant_section(self):
        """Test initialization with empty qdrant configuration section."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = VectorDBInitializer(config_path=config_file.name)
                assert initializer.config["qdrant"] == {}
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_connect_with_missing_config_values(self, mock_qdrant_client):
        """Test connection with missing configuration values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {}}  # No host/port specified
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                result = await initializer.connect()

                # Should use default values
                assert result is True
                mock_qdrant_client.assert_called_with(url="http://localhost:6333")
            finally:
                os.unlink(config_file.name)

    @patch("storage.qdrant_client.QdrantClient")
    async def test_create_collection_missing_config(self, mock_qdrant_client):
        """Test collection creation with missing collection configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"qdrant": {"host": "localhost"}}  # No collection_name or dimensions
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_client = AsyncMock()
                mock_client.get_collection.side_effect = Exception("Collection not found")
                mock_qdrant_client.return_value = mock_client

                initializer = VectorDBInitializer(config_path=config_file.name)
                await initializer.connect()
                result = initializer.create_collection()

                # Should use default values and still succeed
                assert result is True
                mock_client.create_collection.assert_called_once_with()
            finally:
                os.unlink(config_file.name)


@patch("storage.qdrant_client.VectorDBInitializer")
def test_main_function(mock_initializer):
    """Test the main function."""
    from src.storage.qdrant_client import main

    mock_instance = AsyncMock()
    mock_instance.initialize.return_value = True
    mock_initializer.return_value = mock_instance

    # Test successful initialization
    with patch("sys.argv", ["qdrant_client.py"]):
        main()

    mock_instance.initialize.assert_called_once_with()


@patch("storage.qdrant_client.VectorDBInitializer")
def test_main_function_failure(mock_initializer):
    """Test the main function with initialization failure."""
    from src.storage.qdrant_client import main

    mock_instance = AsyncMock()
    mock_instance.initialize.return_value = False
    mock_initializer.return_value = mock_instance

    # Test failed initialization
    with patch("sys.argv", ["qdrant_client.py"]):
        with pytest.raises(SystemExit):
            main()


@patch("storage.qdrant_client.VectorDBInitializer")
def test_main_function_with_exception(mock_initializer):
    """Test the main function with exception during initialization."""
    from src.storage.qdrant_client import main

    mock_initializer.side_effect = Exception("Initialization error")

    # Test initialization with exception
    with patch("sys.argv", ["qdrant_client.py"]):
        with pytest.raises(SystemExit):
            main()
