#!/usr/bin/env python3
"""
Deep tests for Neo4j client to achieve 35% coverage.

This test suite covers:
- Neo4jInitializer initialization and configuration
- Connection management and authentication
- Constraint and index creation
- Configuration loading and error handling
- SSL and security configurations
- Database operations and session management
"""

from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from src.storage.neo4j_client import Neo4jInitializer


class TestNeo4jInitializerInitialization:
    """Test Neo4jInitializer initialization and configuration."""

    def test_init_with_config_injection(self):
        """Test initialization with injected configuration."""
        config = {"neo4j": {"host": "localhost", "port": 7687, "database": "test_graph"}}

        initializer = Neo4jInitializer(config=config, test_mode=True)

        assert initializer.config == config
        assert initializer.test_mode is True
        assert initializer.driver is None
        assert initializer.database == "test_graph"

    def test_init_without_config_test_mode(self):
        """Test initialization without config in test mode."""
        with patch.object(Neo4jInitializer, "_load_config") as mock_load:
            mock_load.return_value = {"neo4j": {"database": "context_graph"}}

            initializer = Neo4jInitializer(config_path="test.yaml", test_mode=True)

            assert initializer.test_mode is True
            assert initializer.database == "context_graph"
            mock_load.assert_called_once_with("test.yaml")

    def test_init_default_database(self):
        """Test initialization with default database name."""
        config = {"neo4j": {}}

        initializer = Neo4jInitializer(config=config)

        assert initializer.database == "context_graph"

    def test_init_empty_neo4j_config(self):
        """Test initialization with empty neo4j config."""
        config = {}

        initializer = Neo4jInitializer(config=config)

        assert initializer.database == "context_graph"


class TestNeo4jInitializerConfigLoading:
    """Test configuration loading methods."""

    def test_load_config_success(self):
        """Test successful configuration loading."""
        config_data = {"neo4j": {"host": "localhost", "port": 7687, "database": "test_db"}}

        with patch("builtins.open", mock_open(read_data=yaml.dump(config_data))):
            with patch("yaml.safe_load", return_value=config_data):
                initializer = Neo4jInitializer.__new__(Neo4jInitializer)
                initializer.test_mode = False
                result = initializer._load_config("test.yaml")

                assert result == config_data

    def test_load_config_file_not_found_test_mode(self):
        """Test config loading when file not found in test mode."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("storage.neo4j_client.get_test_config") as mock_test_config:
                mock_test_config.return_value = {"test": "config"}

                initializer = Neo4jInitializer.__new__(Neo4jInitializer)
                initializer.test_mode = True
                result = initializer._load_config("nonexistent.yaml")

                assert result == {"test": "config"}
                mock_test_config.assert_called_once()

    def test_load_config_file_not_found_production_mode(self):
        """Test config loading when file not found in production mode."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("click.echo") as mock_echo:
                with patch("sys.exit") as mock_exit:
                    initializer = Neo4jInitializer.__new__(Neo4jInitializer)
                    initializer.test_mode = False

                    initializer._load_config("nonexistent.yaml")

                    mock_echo.assert_called_once()
                    mock_exit.assert_called_once_with(1)

    def test_load_config_yaml_error_test_mode(self):
        """Test config loading with YAML error in test mode."""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content")):
            with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
                with patch("storage.neo4j_client.get_test_config") as mock_test_config:
                    mock_test_config.return_value = {"test": "config"}

                    initializer = Neo4jInitializer.__new__(Neo4jInitializer)
                    initializer.test_mode = True
                    result = initializer._load_config("invalid.yaml")

                    assert result == {"test": "config"}

    def test_load_config_yaml_error_production_mode(self):
        """Test config loading with YAML error in production mode."""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content")):
            with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
                from src.core.config_error import ConfigParseError

                initializer = Neo4jInitializer.__new__(Neo4jInitializer)
                initializer.test_mode = False

                with pytest.raises(ConfigParseError):
                    initializer._load_config("invalid.yaml")

    def test_load_config_non_dict_test_mode(self):
        """Test config loading when result is not a dictionary in test mode."""
        with patch("builtins.open", mock_open(read_data="not_a_dict")):
            with patch("yaml.safe_load", return_value="not_a_dict"):
                with patch("storage.neo4j_client.get_test_config") as mock_test_config:
                    mock_test_config.return_value = {"test": "config"}

                    initializer = Neo4jInitializer.__new__(Neo4jInitializer)
                    initializer.test_mode = True
                    result = initializer._load_config("invalid.yaml")

                    assert result == {"test": "config"}

    def test_load_config_non_dict_production_mode(self):
        """Test config loading when result is not a dictionary in production mode."""
        with patch("builtins.open", mock_open(read_data="not_a_dict")):
            with patch("yaml.safe_load", return_value="not_a_dict"):
                from src.core.config_error import ConfigParseError

                initializer = Neo4jInitializer.__new__(Neo4jInitializer)
                initializer.test_mode = False

                with pytest.raises(ConfigParseError):
                    initializer._load_config("invalid.yaml")


class TestNeo4jInitializerConnection:
    """Test Neo4j connection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {"neo4j": {"host": "localhost", "port": 7687, "database": "test_db"}}
        self.initializer = Neo4jInitializer(config=config, test_mode=True)

    def test_connect_success(self):
        """Test successful connection to Neo4j."""
        with patch("storage.neo4j_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

            with patch("neo4j.GraphDatabase.driver") as mock_driver:
                mock_driver_instance = Mock()
                mock_session = Mock()
                mock_result = Mock()
                mock_result.single.return_value = None
                mock_session.run.return_value = mock_result
                mock_driver_instance.session.return_value.__enter__.return_value = mock_session
                mock_driver_instance.session.return_value.__exit__.return_value = None
                mock_driver.return_value = mock_driver_instance

                with patch("click.echo") as mock_echo:
                    result = self.initializer.connect(username="neo4j", password="test_password")

                    assert result is True
                    assert self.initializer.driver == mock_driver_instance
                    mock_driver.assert_called_once_with(
                        "bolt://localhost:7687", auth=("neo4j", "test_password")
                    )
                    mock_echo.assert_called_once()

    def test_connect_with_ssl(self):
        """Test connection with SSL enabled."""
        with patch("storage.neo4j_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "neo4j.example.com", "port": 7687, "ssl": True}

            with patch("neo4j.GraphDatabase.driver") as mock_driver:
                mock_driver_instance = Mock()
                mock_session = Mock()
                mock_result = Mock()
                mock_result.single.return_value = None
                mock_session.run.return_value = mock_result
                mock_driver_instance.session.return_value.__enter__.return_value = mock_session
                mock_driver_instance.session.return_value.__exit__.return_value = None
                mock_driver.return_value = mock_driver_instance

                with patch("click.echo"):
                    result = self.initializer.connect(username="neo4j", password="test_password")

                    assert result is True
                    mock_driver.assert_called_once_with(
                        "bolt+s://neo4j.example.com:7687", auth=("neo4j", "test_password")
                    )

    def test_connect_no_password_prompt(self):
        """Test connection when no password provided (should prompt)."""
        with patch("storage.neo4j_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

            with patch("click.echo") as mock_echo:
                with patch("getpass.getpass", return_value="prompted_password") as mock_getpass:
                    with patch("neo4j.GraphDatabase.driver") as mock_driver:
                        mock_driver_instance = Mock()
                        mock_session = Mock()
                        mock_result = Mock()
                        mock_result.single.return_value = None
                        mock_session.run.return_value = mock_result
                        mock_driver_instance.session.return_value.__enter__.return_value = (
                            mock_session
                        )
                        mock_driver_instance.session.return_value.__exit__.return_value = None
                        mock_driver.return_value = mock_driver_instance

                        result = self.initializer.connect(username="neo4j")

                        assert result is True
                        mock_echo.assert_called()
                        mock_getpass.assert_called_once_with("Password: ")
                        mock_driver.assert_called_once_with(
                            "bolt://localhost:7687", auth=("neo4j", "prompted_password")
                        )

    def test_connect_service_unavailable(self):
        """Test connection when Neo4j service is unavailable."""
        with patch("storage.neo4j_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

            with patch("neo4j.GraphDatabase.driver", side_effect=Exception("Service unavailable")):
                # Mock the ServiceUnavailable exception path by checking in connect method
                from neo4j.exceptions import ServiceUnavailable

                with patch(
                    "neo4j.GraphDatabase.driver",
                    side_effect=ServiceUnavailable("Service unavailable"),
                ):
                    with patch("click.echo") as mock_echo:
                        result = self.initializer.connect(
                            username="neo4j", password="test_password"
                        )

                        assert result is False
                        assert self.initializer.driver is None
                        # Should have called echo multiple times for error messages
                        assert mock_echo.call_count >= 1

    def test_connect_auth_error(self):
        """Test connection with authentication error."""
        with patch("storage.neo4j_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

            from neo4j.exceptions import AuthError

            with patch("neo4j.GraphDatabase.driver", side_effect=AuthError("Invalid credentials")):
                with patch("click.echo") as mock_echo:
                    result = self.initializer.connect(username="neo4j", password="wrong_password")

                    assert result is False
                    assert self.initializer.driver is None
                    mock_echo.assert_called_with("✗ Authentication failed", err=True)

    def test_connect_general_exception(self):
        """Test connection with general exception."""
        with patch("storage.neo4j_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

            with patch("neo4j.GraphDatabase.driver", side_effect=Exception("General error")):
                with patch("storage.neo4j_client.sanitize_error_message") as mock_sanitize:
                    mock_sanitize.return_value = "Sanitized error message"
                    with patch("click.echo") as mock_echo:
                        result = self.initializer.connect(
                            username="neo4j", password="test_password"
                        )

                        assert result is False
                        assert self.initializer.driver is None
                        mock_sanitize.assert_called_once_with(
                            "General error", ["test_password", "neo4j"]
                        )
                        mock_echo.assert_called_with(
                            "✗ Failed to connect: Sanitized error message", err=True
                        )


class TestNeo4jInitializerConstraints:
    """Test constraint creation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {"neo4j": {"database": "test_db"}}
        self.initializer = Neo4jInitializer(config=config, test_mode=True)

    def test_create_constraints_success(self):
        """Test successful constraint creation."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        self.initializer.driver = mock_driver

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_constraints()

            assert result is True
            # Should have called session.run for each constraint
            assert mock_session.run.call_count > 0
            # Should have echoed success messages
            assert mock_echo.call_count > 0

    def test_create_constraints_no_driver(self):
        """Test constraint creation when not connected."""
        self.initializer.driver = None

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_constraints()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_create_constraints_exception(self):
        """Test constraint creation with exception."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.run.side_effect = Exception("Constraint error")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        self.initializer.driver = mock_driver

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_constraints()

            assert result is False
            # Should have called echo with error message
            error_calls = [call for call in mock_echo.call_args_list if call[0][0].startswith("✗")]
            assert len(error_calls) > 0


class TestNeo4jInitializerIndexes:
    """Test index creation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {"neo4j": {"database": "test_db"}}
        self.initializer = Neo4jInitializer(config=config, test_mode=True)

    def test_create_indexes_success(self):
        """Test successful index creation."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        self.initializer.driver = mock_driver

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_indexes()

            assert result is True
            # Should have called session.run for each index
            assert mock_session.run.call_count > 0
            # Should have echoed success messages
            assert mock_echo.call_count > 0

    def test_create_indexes_no_driver(self):
        """Test index creation when not connected."""
        self.initializer.driver = None

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_indexes()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_create_indexes_exception(self):
        """Test index creation with exception."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.run.side_effect = Exception("Index error")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        self.initializer.driver = mock_driver

        with patch("click.echo") as mock_echo:
            result = self.initializer.create_indexes()

            assert result is False
            # Should have called echo with error message
            error_calls = [call for call in mock_echo.call_args_list if call[0][0].startswith("✗")]
            assert len(error_calls) > 0


class TestNeo4jInitializerIntegration:
    """Test integration scenarios and edge cases."""

    def test_full_initialization_workflow(self):
        """Test complete initialization workflow."""
        config = {"neo4j": {"host": "localhost", "port": 7687, "database": "integration_test"}}

        initializer = Neo4jInitializer(config=config, test_mode=True)

        # Verify initialization
        assert initializer.config == config
        assert initializer.database == "integration_test"
        assert initializer.driver is None

        # Test connection (mocked)
        with patch("storage.neo4j_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

            with patch("neo4j.GraphDatabase.driver") as mock_driver:
                mock_driver_instance = Mock()
                mock_session = Mock()
                mock_result = Mock()
                mock_result.single.return_value = None
                mock_session.run.return_value = mock_result
                mock_driver_instance.session.return_value.__enter__.return_value = mock_session
                mock_driver_instance.session.return_value.__exit__.return_value = None
                mock_driver.return_value = mock_driver_instance

                with patch("click.echo"):
                    connection_result = initializer.connect(password="test_password")

                    assert connection_result is True
                    assert initializer.driver is not None

    def test_error_handling_chain(self):
        """Test error handling in method chain."""
        initializer = Neo4jInitializer(config={}, test_mode=True)

        # Test constraint creation without connection
        with patch("click.echo"):
            constraint_result = initializer.create_constraints()
            assert constraint_result is False

        # Test index creation without connection
        with patch("click.echo"):
            index_result = initializer.create_indexes()
            assert index_result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
