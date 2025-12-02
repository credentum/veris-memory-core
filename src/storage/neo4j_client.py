#!/usr/bin/env python3
"""
neo4j_client.py: Neo4j client for context storage system

This module:
1. Provides Neo4j connection management
2. Creates constraints and indexes
3. Sets up node labels and relationship types
4. Manages graph database operations
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import yaml
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from neo4j.time import DateTime, Date, Time, Duration

logger = logging.getLogger(__name__)

# Import types for protocol compliance
try:
    from .types import JSON, NodeID, QueryResult
except ImportError:
    from types import JSON, NodeID, QueryResult

# Import configuration error handling
try:
    from ..core.config_error import ConfigFileNotFoundError, ConfigParseError
    from ..core.test_config import get_test_config
except ImportError:
    from core.config_error import ConfigFileNotFoundError, ConfigParseError
    from core.test_config import get_test_config


def serialize_neo4j_value(value: Any) -> Any:
    """Convert Neo4j-specific types to JSON-serializable Python types.

    Args:
        value: Value from Neo4j query result

    Returns:
        JSON-serializable version of the value
    """
    # Handle Neo4j temporal types
    if isinstance(value, (DateTime, Date, Time)):
        return value.iso_format()
    elif isinstance(value, Duration):
        # Convert duration to total seconds for simplicity
        return {
            "months": value.months,
            "days": value.days,
            "seconds": value.seconds,
            "nanoseconds": value.nanoseconds
        }
    # Handle lists (may contain temporal types)
    elif isinstance(value, list):
        return [serialize_neo4j_value(item) for item in value]
    # Handle dictionaries (may contain temporal types)
    elif isinstance(value, dict):
        return {k: serialize_neo4j_value(v) for k, v in value.items()}
    # Handle objects with __dict__ (Neo4j nodes/relationships)
    elif hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool)):
        try:
            return {k: serialize_neo4j_value(v) for k, v in dict(value).items()}
        except (TypeError, AttributeError):
            # If conversion fails, return string representation
            return str(value)
    # Return primitive types as-is
    else:
        return value


class Neo4jInitializer:
    """Initialize and configure Neo4j graph database"""

    def __init__(
        self,
        config_path: str = ".ctxrc.yaml",
        config: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ):
        """Initialize Neo4j client with optional config injection for testing.

        Args:
            config_path: Path to configuration file
            config: Optional configuration dictionary (overrides file loading)
            test_mode: If True, use test defaults when config is missing
        """
        self.test_mode = test_mode

        if config is not None:
            self.config = config
        else:
            self.config = self._load_config(config_path)

        self.driver: Optional[Driver] = None
        self.database = self.config.get("neo4j", {}).get("database", "context_graph")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from .ctxrc.yaml"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    if self.test_mode:
                        return get_test_config()
                    raise ConfigParseError(config_path, "Configuration must be a dictionary")
                return config
        except FileNotFoundError:
            if self.test_mode:
                # Return test configuration when in test mode
                return get_test_config()
            # In production, still use sys.exit for backward compatibility
            click.echo(f"Error: {config_path} not found", err=True)
            sys.exit(1)
        except yaml.YAMLError as e:
            if self.test_mode:
                return get_test_config()
            raise ConfigParseError(config_path, str(e))

    def connect(self, username: str = "neo4j", password: Optional[str] = None) -> bool:
        """Connect to Neo4j instance"""
        # Import locally
        from ..core.utils import get_secure_connection_config

        neo4j_config = get_secure_connection_config(self.config, "neo4j")
        host = neo4j_config["host"]
        port = neo4j_config.get("port", 7687)
        use_ssl = neo4j_config.get("ssl", False)

        # Use appropriate protocol based on SSL setting
        protocol = "bolt+s" if use_ssl else "bolt"
        uri = f"{protocol}://{host}:{port}"

        try:
            if not password:
                click.echo("Neo4j password required")
                import getpass

                password = getpass.getpass("Password: ")

            self.driver = GraphDatabase.driver(uri, auth=(username, password))

            # Test connection
            if self.driver:
                with self.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    result.single()

            click.echo(f"✓ Connected to Neo4j at {uri}")
            return True

        except ServiceUnavailable:
            click.echo(f"✗ Neo4j is not available at {uri}", err=True)
            click.echo("Please ensure Neo4j is running:")
            click.echo(
                "  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5"
            )
            return False
        except AuthError:
            click.echo("✗ Authentication failed", err=True)
            return False
        except Exception as e:
            # Import locally to avoid circular imports
            from ..core.utils import sanitize_error_message

            # Sanitize error message to remove potential passwords
            sensitive_values = []
            if password:
                sensitive_values.append(password)
            if username:
                sensitive_values.append(username)
            error_msg = sanitize_error_message(str(e), sensitive_values)
            click.echo(f"✗ Failed to connect: {error_msg}", err=True)
            return False

    def create_constraints(self) -> bool:
        """Create uniqueness constraints for node types"""
        constraints = [
            # Document nodes
            ("Document", "id"),
            ("Design", "id"),
            ("Decision", "id"),
            ("Sprint", "id"),
            # System nodes
            ("Agent", "name"),
            ("Phase", "number"),
            ("Task", "id"),
            ("Metric", "name"),
            # Version tracking
            ("Version", "hash"),
        ]

        try:
            if not self.driver:
                click.echo("✗ Not connected to Neo4j", err=True)
                return False

            with self.driver.session(database=self.database) as session:
                for label, property in constraints:
                    query = (
                        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                        f"REQUIRE n.{property} IS UNIQUE"
                    )
                    session.run(query)
                    click.echo(f"  Created constraint: {label}.{property}")

            return True
        except Exception as e:
            click.echo(f"✗ Failed to create constraints: {e}", err=True)
            return False

    def create_indexes(self) -> bool:
        """Create indexes for common queries"""
        indexes: List[Union[Tuple[str, List[str]], Tuple[str, List[str], str]]] = [
            # Document indexes
            ("Document", ["document_type", "created_date"]),
            ("Document", ["last_modified"]),
            ("Document", ["status"]),
            # Relationship traversal
            ("Sprint", ["sprint_number", "status"]),
            ("Task", ["status", "assigned_to"]),
            # Full-text search
            ("Document", ["title"], "fulltext"),
            ("Document", ["description"], "fulltext"),
        ]

        try:
            if not self.driver:
                click.echo("✗ Not connected to Neo4j", err=True)
                return False

            with self.driver.session(database=self.database) as session:
                for index_spec in indexes:
                    if len(index_spec) == 2:
                        label, properties = index_spec
                        index_type = "btree"
                    else:
                        label, properties, index_type = index_spec

                    props_str = ", ".join([f"n.{p}" for p in properties])
                    index_name = f"idx_{label}_{'_'.join(properties)}"

                    if index_type == "fulltext":
                        query = f"""
                        CALL db.index.fulltext.createNodeIndex(
                            '{index_name}',
                            ['{label}'],
                            {properties}
                        )
                        """
                    else:
                        query = (
                            f"CREATE INDEX {index_name} IF NOT EXISTS "
                            f"FOR (n:{label}) ON ({props_str})"
                        )

                    try:
                        session.run(query)
                        click.echo(f"  Created index: {index_name}")
                    except Exception as e:
                        if "already exists" not in str(e):
                            click.echo(f"  Warning: {e}")

            return True
        except Exception as e:
            click.echo(f"✗ Failed to create indexes: {e}", err=True)
            return False

    def setup_graph_schema(self) -> bool:
        """Initialize the graph schema with example relationships"""
        try:
            if not self.driver:
                click.echo("✗ Not connected to Neo4j", err=True)
                return False

            with self.driver.session(database=self.database) as session:
                # Create root System node
                session.run(
                    """
                    MERGE (s:System {id: 'agent-context-system'})
                    SET s.name = 'Agent-First Context System',
                        s.version = $version,
                        s.created_date = $date
                """,
                    version=self.config.get("system", {}).get("schema_version", "1.0.0"),
                    date=self.config.get("system", {}).get("created_date", "2025-07-11"),
                )

                # Create agent nodes
                agents = [
                    ("code_agent", "Code Agent", "Primary implementation agent"),
                    ("doc_agent", "Documentation Agent", "Maintains documentation"),
                    ("pm_agent", "Project Manager Agent", "Manages sprints and tasks"),
                    ("ci_agent", "CI/CD Agent", "Handles testing and deployment"),
                ]

                for agent_id, name, description in agents:
                    session.run(
                        """
                        MERGE (a:Agent {name: $name})
                        SET a.id = $id,
                            a.description = $description,
                            a.active = true
                    """,
                        id=agent_id,
                        name=name,
                        description=description,
                    )

                # Link agents to system
                session.run(
                    """
                    MATCH (s:System {id: 'agent-context-system'})
                    MATCH (a:Agent)
                    MERGE (s)-[:HAS_AGENT]->(a)
                """
                )

                # Create document type hierarchy
                doc_types = [
                    ("design", "Design Document", "DEFINES_ARCHITECTURE"),
                    ("decision", "Decision Record", "RECORDS_DECISION"),
                    ("sprint", "Sprint Plan", "TRACKS_PROGRESS"),
                ]

                for type_id, type_name, rel_type in doc_types:
                    # Create document type node with parameterized query
                    session.run(
                        """
                        MERGE (dt:DocumentType {id: $type_id})
                        SET dt.name = $type_name
                    """,
                        type_id=type_id,
                        type_name=type_name,
                    )

                    # Create relationship - use a generic relationship with type property
                    # This avoids dynamic query construction
                    session.run(
                        """
                        MATCH (s:System {id: 'agent-context-system'})
                        MATCH (dt:DocumentType {id: $type_id})
                        MERGE (s)-[r:HAS_DOCUMENT_TYPE]->(dt)
                        SET r.semantic_type = $rel_type
                    """,
                        type_id=type_id,
                        rel_type=rel_type,
                    )

                click.echo("✓ Graph schema initialized")
                return True

        except Exception as e:
            click.echo(f"✗ Failed to setup schema: {e}", err=True)
            return False

    def verify_setup(self) -> bool:
        """Verify the Neo4j setup"""
        if not self.driver:
            click.echo("✗ Not connected to Neo4j", err=True)
            return False

        try:
            with self.driver.session(database=self.database) as session:
                # Count nodes by label
                result = session.run(
                    """
                    CALL db.labels() YIELD label
                    CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
                    YIELD value
                    RETURN label, value.count as count
                    ORDER BY label
                """
                )

                click.echo("\nNode counts by label:")
                for record in result:
                    click.echo(f"  {record['label']}: {record['count']}")

                # Count relationships
                result = session.run(
                    """
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                    ORDER BY type
                """
                )

                click.echo("\nRelationship counts by type:")
                for record in result:
                    click.echo(f"  {record['type']}: {record['count']}")

                return True

        except Exception:
            # APOC might not be installed, try simpler query
            try:
                if not self.driver:
                    return False

                with self.driver.session(database=self.database) as session:
                    result = session.run("MATCH (n) RETURN count(n) as total")
                    node_record = result.single()
                    if node_record:
                        total = node_record["total"]
                        click.echo(f"\nTotal nodes: {total}")

                    result = session.run("MATCH ()-[r]->() RETURN count(r) as total")
                    rel_record = result.single()
                    if rel_record:
                        total = rel_record["total"]
                        click.echo(f"Total relationships: {total}")

                return True
            except Exception as e2:
                click.echo(f"✗ Failed to verify setup: {e2}", err=True)
                return False

    def close(self) -> None:
        """Close the driver connection"""
        if self.driver:
            self.driver.close()

    def create_node(self, labels: List[str], properties: JSON) -> NodeID:
        """Create a graph node.

        Args:
            labels: List of node labels
            properties: Node properties as JSON

        Returns:
            NodeID: The ID of the created node

        Raises:
            RuntimeError: If not connected to Neo4j or creation fails
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")

        try:
            # Validate labels with strict security rules
            if labels:
                import re

                # Only allow alphanumeric characters, underscore, and hyphen
                # Must start with a letter, no consecutive special chars
                label_pattern = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")

                validated_labels = []
                for label in labels:
                    if not isinstance(label, str):
                        raise ValueError(f"Label must be string, got {type(label).__name__}")
                    if not (1 <= len(label) <= 50):  # Reasonable length limits
                        raise ValueError(f"Label length must be 1-50 characters, got {len(label)}")
                    if not label_pattern.match(label):
                        raise ValueError(
                            f"Invalid label format: '{label}'. Must start with letter and contain only letters, numbers, _ or -"
                        )
                    if "__" in label or "--" in label:  # No consecutive special chars
                        raise ValueError(
                            f"Invalid label format: '{label}'. No consecutive special characters allowed"
                        )
                    validated_labels.append(label)

                labels_str = ":".join(validated_labels)
            else:
                labels_str = "Node"

            # Create the node - labels must be validated but cannot be parameterized in Cypher
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    f"CREATE (n:{labels_str}) SET n = $properties RETURN id(n) as node_id",
                    properties=properties or {},
                )
                record = result.single()
                if record:
                    return str(record["node_id"])
                else:
                    raise RuntimeError("Failed to create node - no result returned")

        except Exception as e:
            raise RuntimeError(f"Failed to create node: {e}")

    def query(self, cypher: str, parameters: Optional[JSON] = None) -> QueryResult:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string
            parameters: Query parameters

        Returns:
            QueryResult: Query results as a list of dictionaries

        Raises:
            RuntimeError: If not connected to Neo4j or query fails
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")

        # Retry logic for "Session is closed" errors
        # This handles cleanup scenarios where sessions may be closed prematurely
        max_retries = 2
        for attempt in range(max_retries):
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, parameters or {})
                    records = []
                    for record in result:
                        # Convert neo4j record to dictionary with proper serialization
                        record_dict = {}
                        for key in record.keys():
                            value = record[key]
                            # Use serialization helper to handle all Neo4j types
                            record_dict[key] = serialize_neo4j_value(value)
                        records.append(record_dict)
                    return records

            except Exception as e:
                error_msg = str(e).lower()
                # Check if this is a "session is closed" error
                if "session" in error_msg and ("closed" in error_msg or "invalid" in error_msg):
                    if attempt < max_retries - 1:
                        logger.warning(f"Neo4j session closed, retrying with fresh session (attempt {attempt + 1}/{max_retries})")
                        continue  # Retry with fresh session
                    else:
                        logger.error(f"Neo4j session still closed after {max_retries} attempts")
                # Re-raise other errors or final retry failure
                raise RuntimeError(f"Failed to execute query: {e}")

    def create_relationship(
        self,
        start_node: NodeID,
        end_node: NodeID,
        relationship_type: str,
        properties: Optional[JSON] = None,
    ) -> str:
        """Create a relationship between nodes.

        Args:
            start_node: ID of the start node
            end_node: ID of the end node
            relationship_type: Type of relationship
            properties: Optional relationship properties

        Returns:
            str: The ID of the created relationship

        Raises:
            RuntimeError: If not connected to Neo4j or creation fails
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")

        # Validate node IDs
        try:
            start_id = int(start_node)
            end_id = int(end_node)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid node ID format - Must be numeric strings: {e}")

        # Validate relationship type
        if not relationship_type or not isinstance(relationship_type, str):
            raise ValueError("Relationship type must be a non-empty string")

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (a), (b)
                    WHERE id(a) = $start_id AND id(b) = $end_id
                    CREATE (a)-[r:TYPE]->(b)
                    SET r = $properties, r.type = $rel_type
                    RETURN id(r) as rel_id
                    """,
                    start_id=start_id,
                    end_id=end_id,
                    rel_type=relationship_type,
                    properties=properties or {},
                )
                record = result.single()
                if record:
                    return str(record["rel_id"])
                else:
                    raise RuntimeError("Failed to create relationship - nodes may not exist")

        except Exception as e:
            raise RuntimeError(f"Failed to create relationship: {e}")


@click.command()
@click.option("--username", default="neo4j", help="Neo4j username")
@click.option("--password", help="Neo4j password (will prompt if not provided)")
@click.option("--skip-constraints", is_flag=True, help="Skip constraint creation")
@click.option("--skip-indexes", is_flag=True, help="Skip index creation")
@click.option("--skip-schema", is_flag=True, help="Skip schema initialization")
def main(
    username: str,
    password: Optional[str],
    skip_constraints: bool,
    skip_indexes: bool,
    skip_schema: bool,
) -> None:
    """Initialize Neo4j graph database for the Agent-First Context System"""

    click.echo("=== Neo4j Graph Database Initialization ===\n")

    initializer = Neo4jInitializer()

    try:
        # Connect to Neo4j
        if not initializer.connect(username=username, password=password):
            sys.exit(1)

        # Create database if needed (Enterprise feature - skip for Community Edition)
        if not initializer.driver:
            click.echo("✗ Not connected to Neo4j", err=True)
            sys.exit(1)

        # Check if we're using Enterprise Edition
        try:
            with initializer.driver.session(database="system") as session:
                db_name = initializer.database
                result = session.run("SHOW DATABASES WHERE name = $name", name=db_name)
                if not result.single():
                    click.echo(f"Creating database '{db_name}'...")
                    session.run(f"CREATE DATABASE {db_name}")
        except Exception as e:
            if "UnsupportedAdministrationCommand" in str(e):
                click.echo("Note: Using Neo4j Community Edition - using default database")
                # Use default database for Community Edition
                initializer.database = "neo4j"
            else:
                raise e

        # Create constraints
        if not skip_constraints:
            click.echo("\nCreating constraints...")
            if not initializer.create_constraints():
                sys.exit(1)

        # Create indexes
        if not skip_indexes:
            click.echo("\nCreating indexes...")
            if not initializer.create_indexes():
                sys.exit(1)

        # Setup schema
        if not skip_schema:
            click.echo("\nSetting up graph schema...")
            if not initializer.setup_graph_schema():
                sys.exit(1)

        # Verify
        click.echo("\nVerifying setup...")
        initializer.verify_setup()

        click.echo("\n✓ Neo4j initialization complete!")

    finally:
        initializer.close()


if __name__ == "__main__":
    main()
