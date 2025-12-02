#!/usr/bin/env python3
"""
Comprehensive tests for the Neo4j client to increase coverage.

This test suite covers graph database initialization and operations
to achieve high code coverage for the storage.neo4j_client module.
"""

import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from src.storage.neo4j_client import Neo4jInitializer  # noqa: E402


class TestNeo4jInitializer:
    """Test suite for Neo4jInitializer class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "database": "neo4j",
                    "username": "neo4j",
                    "password": "password",
                }
            }
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = Neo4jInitializer(config_path=config_file.name)
                assert initializer.config == config
                assert initializer.driver is None
                assert initializer.database == "neo4j"
            finally:
                os.unlink(config_file.name)

    def test_init_missing_config_file(self):
        """Test initialization with missing configuration file."""
        with pytest.raises(SystemExit):
            Neo4jInitializer(config_path="/nonexistent/config.yaml")

    def test_init_invalid_yaml(self):
        """Test initialization with invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config_file.write("invalid: yaml: content: [")
            config_file.flush()

            try:
                with pytest.raises(SystemExit):
                    Neo4jInitializer(config_path=config_file.name)
            finally:
                os.unlink(config_file.name)

    def test_init_non_dict_config(self):
        """Test initialization with non-dictionary configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            yaml.dump(["not", "a", "dict"], config_file)
            config_file.flush()

            try:
                with pytest.raises(SystemExit):
                    Neo4jInitializer(config_path=config_file.name)
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_connect_success_basic(self, mock_driver):
        """Test successful connection with basic credentials."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687", "database": "neo4j"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_driver_instance = AsyncMock()
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                result = initializer.connect(username="test_user", password="test_pass")

                assert result is True
                assert initializer.driver == mock_driver_instance
                mock_driver.assert_called_with(
                    "bolt://localhost:7687", auth=("test_user", "test_pass")
                )
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_connect_success_with_ssl(self, mock_driver):
        """Test successful connection with SSL configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687", "database": "neo4j"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_driver_instance = AsyncMock()
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                result = initializer.connect(
                    username="test_user",
                    password="test_pass",
                    encrypted=True,
                    trust="TRUST_ALL_CERTIFICATES",
                )

                assert result is True
                mock_driver.assert_called_with(
                    "bolt://localhost:7687",
                    auth=("test_user", "test_pass"),
                    encrypted=True,
                    trust="TRUST_ALL_CERTIFICATES",
                )
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_connect_with_custom_uri_from_env(self, mock_driver):
        """Test connection with URI from environment variable."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"database": "test_db"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_driver_instance = AsyncMock()
                mock_driver.return_value = mock_driver_instance

                with patch.dict(os.environ, {"NEO4J_URI": "bolt://env:7687"}):
                    initializer = Neo4jInitializer(config_path=config_file.name)
                    result = initializer.connect(username="user", password="pass")

                assert result is True
                mock_driver.assert_called_with("bolt://env:7687", auth=("user", "pass"))
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_connect_failure(self, mock_driver):
        """Test connection failure handling."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_driver.side_effect = Exception("Connection failed")

                initializer = Neo4jInitializer(config_path=config_file.name)
                result = initializer.connect(username="user", password="pass")

                assert result is False
                assert initializer.driver is None
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    async def test_verify_connection_success(self, mock_driver):
        """Test successful connection verification."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_session = AsyncMock()
                mock_session.run.return_value.single.return_value = {"result": 1}
                mock_driver_instance = AsyncMock()
                mock_driver_instance.session.return_value.__enter__ = Mock(
                    return_value=mock_session
                )
                mock_driver_instance.session.return_value.__exit__ = Mock(return_value=None)
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                initializer.connect(username="user", password="pass")
                result = await initializer.verify_connection()

                assert result is True
                mock_session.run.assert_called_with("RETURN 1 as result")
            finally:
                os.unlink(config_file.name)

    async def test_verify_connection_no_driver(self):
        """Test connection verification without driver."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = Neo4jInitializer(config_path=config_file.name)
                result = await initializer.verify_connection()

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    async def test_verify_connection_failure(self, mock_driver):
        """Test connection verification failure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_session = AsyncMock()
                mock_session.run.side_effect = Exception("Query failed")
                mock_driver_instance = AsyncMock()
                mock_driver_instance.session.return_value.__enter__ = Mock(
                    return_value=mock_session
                )
                mock_driver_instance.session.return_value.__exit__ = Mock(return_value=None)
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                initializer.connect(username="user", password="pass")
                result = await initializer.verify_connection()

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_create_indexes_success(self, mock_driver):
        """Test successful index creation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687", "database": "test_db"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_session = AsyncMock()
                mock_driver_instance = AsyncMock()
                mock_driver_instance.session.return_value.__enter__ = Mock(
                    return_value=mock_session
                )
                mock_driver_instance.session.return_value.__exit__ = Mock(return_value=None)
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                initializer.connect(username="user", password="pass")
                result = initializer.create_indexes()

                assert result is True
                # Should call session.run for each index
                assert mock_session.run.call_count >= 3  # At least 3 indexes
            finally:
                os.unlink(config_file.name)

    def test_create_indexes_no_driver(self):
        """Test index creation without driver."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = Neo4jInitializer(config_path=config_file.name)
                result = initializer.create_indexes()

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_create_indexes_failure(self, mock_driver):
        """Test index creation failure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687", "database": "test_db"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_session = AsyncMock()
                mock_session.run.side_effect = Exception("Index creation failed")
                mock_driver_instance = AsyncMock()
                mock_driver_instance.session.return_value.__enter__ = Mock(
                    return_value=mock_session
                )
                mock_driver_instance.session.return_value.__exit__ = Mock(return_value=None)
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                initializer.connect(username="user", password="pass")
                result = initializer.create_indexes()

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_create_constraints_success(self, mock_driver):
        """Test successful constraint creation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687", "database": "test_db"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_session = AsyncMock()
                mock_driver_instance = AsyncMock()
                mock_driver_instance.session.return_value.__enter__ = Mock(
                    return_value=mock_session
                )
                mock_driver_instance.session.return_value.__exit__ = Mock(return_value=None)
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                initializer.connect(username="user", password="pass")
                result = initializer.create_constraints()

                assert result is True
                # Should call session.run for constraints
                assert mock_session.run.call_count >= 1
            finally:
                os.unlink(config_file.name)

    def test_create_constraints_no_driver(self):
        """Test constraint creation without driver."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = Neo4jInitializer(config_path=config_file.name)
                result = initializer.create_constraints()

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_create_constraints_failure(self, mock_driver):
        """Test constraint creation failure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687", "database": "test_db"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_session = AsyncMock()
                mock_session.run.side_effect = Exception("Constraint creation failed")
                mock_driver_instance = AsyncMock()
                mock_driver_instance.session.return_value.__enter__ = Mock(
                    return_value=mock_session
                )
                mock_driver_instance.session.return_value.__exit__ = Mock(return_value=None)
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                initializer.connect(username="user", password="pass")
                result = initializer.create_constraints()

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_initialize_success(self, mock_driver):
        """Test successful full initialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687", "database": "test_db"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_session = AsyncMock()
                mock_session.run.return_value.single.return_value = {"result": 1}
                mock_driver_instance = AsyncMock()
                mock_driver_instance.session.return_value.__enter__ = Mock(
                    return_value=mock_session
                )
                mock_driver_instance.session.return_value.__exit__ = Mock(return_value=None)
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                result = initializer.initialize(username="user", password="pass")

                assert result is True
                assert initializer.driver is not None
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_initialize_connect_failure(self, mock_driver):
        """Test initialization failure during connection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_driver.side_effect = Exception("Connection failed")

                initializer = Neo4jInitializer(config_path=config_file.name)
                result = initializer.initialize(username="user", password="pass")

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_initialize_verification_failure(self, mock_driver):
        """Test initialization failure during verification."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_session = AsyncMock()
                mock_session.run.side_effect = Exception("Verification failed")
                mock_driver_instance = AsyncMock()
                mock_driver_instance.session.return_value.__enter__ = Mock(
                    return_value=mock_session
                )
                mock_driver_instance.session.return_value.__exit__ = Mock(return_value=None)
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                result = initializer.initialize(username="user", password="pass")

                assert result is False
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_close_with_driver(self, mock_driver):
        """Test closing connection with active driver."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_driver_instance = AsyncMock()
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                initializer.connect(username="user", password="pass")
                initializer.close()

                mock_driver_instance.close.assert_called_once_with()
                assert initializer.driver is None
            finally:
                os.unlink(config_file.name)

    def test_close_without_driver(self):
        """Test closing connection without active driver."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = Neo4jInitializer(config_path=config_file.name)
                initializer.close()  # Should not raise exception
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_close_with_exception(self, mock_driver):
        """Test closing connection with exception during close."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_driver_instance = AsyncMock()
                mock_driver_instance.close.side_effect = Exception("Close failed")
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                initializer.connect(username="user", password="pass")
                initializer.close()  # Should not raise exception

                assert initializer.driver is None
            finally:
                os.unlink(config_file.name)


class TestNeo4jConfigHandling:
    """Test suite for configuration handling edge cases."""

    def test_missing_neo4j_section(self):
        """Test initialization with missing neo4j configuration section."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"other_section": {"key": "value"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = Neo4jInitializer(config_path=config_file.name)
                # Should handle missing neo4j section gracefully
                assert "neo4j" not in initializer.config
                assert initializer.database == "neo4j"  # Default database
            finally:
                os.unlink(config_file.name)

    def test_empty_neo4j_section(self):
        """Test initialization with empty neo4j configuration section."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = Neo4jInitializer(config_path=config_file.name)
                assert initializer.config["neo4j"] == {}
                assert initializer.database == "neo4j"  # Default database
            finally:
                os.unlink(config_file.name)

    def test_custom_database_name(self):
        """Test initialization with custom database name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"database": "custom_db"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                initializer = Neo4jInitializer(config_path=config_file.name)
                assert initializer.database == "custom_db"
            finally:
                os.unlink(config_file.name)

    @patch("storage.neo4j_client.GraphDatabase.driver")
    def test_connect_with_all_ssl_options(self, mock_driver):
        """Test connection with all SSL options."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            config = {"neo4j": {"uri": "bolt://localhost:7687"}}
            yaml.dump(config, config_file)
            config_file.flush()

            try:
                mock_driver_instance = AsyncMock()
                mock_driver.return_value = mock_driver_instance

                initializer = Neo4jInitializer(config_path=config_file.name)
                result = initializer.connect(
                    username="user",
                    password="pass",
                    encrypted=True,
                    trust="TRUST_CUSTOM_CA_SIGNED_CERTIFICATES",
                    trusted_certificates="/path/to/ca.crt",
                    client_certificate=("/path/to/cert.pem", "/path/to/key.pem"),
                )

                assert result is True
                mock_driver.assert_called_with(
                    "bolt://localhost:7687",
                    auth=("user", "pass"),
                    encrypted=True,
                    trust="TRUST_CUSTOM_CA_SIGNED_CERTIFICATES",
                    trusted_certificates="/path/to/ca.crt",
                    client_certificate=("/path/to/cert.pem", "/path/to/key.pem"),
                )
            finally:
                os.unlink(config_file.name)


@patch("storage.neo4j_client.Neo4jInitializer")
def test_main_function(mock_initializer):
    """Test the main function."""
    from src.storage.neo4j_client import main

    mock_instance = AsyncMock()
    mock_instance.initialize.return_value = True
    mock_initializer.return_value = mock_instance

    # Test successful initialization
    with patch("sys.argv", ["neo4j_client.py"]):
        with patch.dict(os.environ, {"NEO4J_USERNAME": "user", "NEO4J_PASSWORD": "pass"}):
            main()

    mock_instance.initialize.assert_called_once_with(username="user", password="pass")


@patch("storage.neo4j_client.Neo4jInitializer")
def test_main_function_missing_credentials(mock_initializer):
    """Test the main function with missing credentials."""
    from src.storage.neo4j_client import main

    # Test with missing credentials
    with patch("sys.argv", ["neo4j_client.py"]):
        with pytest.raises(SystemExit):
            main()


@patch("storage.neo4j_client.Neo4jInitializer")
def test_main_function_initialization_failure(mock_initializer):
    """Test the main function with initialization failure."""
    from src.storage.neo4j_client import main

    mock_instance = AsyncMock()
    mock_instance.initialize.return_value = False
    mock_initializer.return_value = mock_instance

    # Test failed initialization
    with patch("sys.argv", ["neo4j_client.py"]):
        with patch.dict(os.environ, {"NEO4J_USERNAME": "user", "NEO4J_PASSWORD": "pass"}):
            with pytest.raises(SystemExit):
                main()


@patch("storage.neo4j_client.Neo4jInitializer")
def test_main_function_with_exception(mock_initializer):
    """Test the main function with exception during initialization."""
    from src.storage.neo4j_client import main

    mock_initializer.side_effect = Exception("Initialization error")

    # Test initialization with exception
    with patch("sys.argv", ["neo4j_client.py"]):
        with patch.dict(os.environ, {"NEO4J_USERNAME": "user", "NEO4J_PASSWORD": "pass"}):
            with pytest.raises(SystemExit):
                main()
