#!/usr/bin/env python3
"""
Neo4j Schema Initialization - Python Implementation

This script handles Neo4j schema initialization using the Python Neo4j driver.
Extracted from init-neo4j-schema.sh for better testability and maintainability.

Usage:
    # Execute a single query
    python3 init_neo4j_schema.py --query "RETURN 1"

    # Execute a Cypher file
    python3 init_neo4j_schema.py --file deployments/neo4j-init/001-init-schema.cypher

Environment Variables:
    NEO4J_HOST: Neo4j host (default: localhost)
    NEO4J_PORT: Neo4j port (default: 7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password (required)
"""

import sys
import os
import argparse
from typing import Optional
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


class Neo4jSchemaInitializer:
    """Handles Neo4j schema initialization operations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        user: str = "neo4j",
        password: str = "",
    ):
        """
        Initialize the Neo4j schema initializer.

        Args:
            host: Neo4j host address
            port: Neo4j port number
            user: Neo4j username
            password: Neo4j password
        """
        if not password:
            raise ValueError("NEO4J_PASSWORD is required")

        self.uri = f"bolt://{host}:{port}"
        self.auth = (user, password)
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✓ Connected to Neo4j at {self.uri}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def execute_query(self, query: str) -> bool:
        """
        Execute a single Cypher query.

        Args:
            query: Cypher query to execute

        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")

        try:
            with self.driver.session() as session:
                session.run(query)
            print(f"✓ Executed: {query[:50]}...")
            return True
        except Neo4jError as e:
            # Check if error is due to "already exists" (idempotent operations)
            if "already exists" in str(e).lower():
                print(f"ℹ Query already applied (idempotent): {query[:50]}...")
                return True
            else:
                print(f"✗ Failed: {e}")
                return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False

    def execute_file(self, file_path: str) -> tuple[int, int]:
        """
        Execute a Cypher file containing multiple statements.

        Args:
            file_path: Path to Cypher file

        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cypher file not found: {file_path}")

        with open(file_path, 'r') as f:
            cypher_script = f.read()

        # Split by semicolon and filter out comments/empty lines
        statements = [
            s.strip()
            for s in cypher_script.split(';')
            if s.strip() and not s.strip().startswith('//')
        ]

        successful = 0
        failed = 0

        print(f"Executing {len(statements)} statements from {file_path}")

        for i, stmt in enumerate(statements, 1):
            if stmt:
                print(f"\n[{i}/{len(statements)}] ", end="")
                if self.execute_query(stmt):
                    successful += 1
                else:
                    failed += 1

        return successful, failed


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Initialize Neo4j schema using Python driver"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Execute a single Cypher query"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Execute a Cypher file with multiple statements"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("NEO4J_HOST", "localhost"),
        help="Neo4j host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("NEO4J_PORT", "7687")),
        help="Neo4j port (default: 7687)"
    )
    parser.add_argument(
        "--user",
        type=str,
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: neo4j)"
    )

    args = parser.parse_args()

    # Get password from environment
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        print("❌ ERROR: NEO4J_PASSWORD environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Require either --query or --file
    if not args.query and not args.file:
        parser.print_help()
        print("\n❌ ERROR: Either --query or --file must be specified", file=sys.stderr)
        sys.exit(1)

    # Initialize and connect
    initializer = Neo4jSchemaInitializer(
        host=args.host,
        port=args.port,
        user=args.user,
        password=password
    )

    try:
        initializer.connect()

        if args.query:
            # Execute single query
            success = initializer.execute_query(args.query)
            exit_code = 0 if success else 1

        elif args.file:
            # Execute file
            successful, failed = initializer.execute_file(args.file)

            print(f"\n📊 Results:")
            print(f"  ✅ Successful: {successful}")
            print(f"  ❌ Failed: {failed}")

            # Exit with error if any statements failed
            exit_code = 0 if failed == 0 else 1

        sys.exit(exit_code)

    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        initializer.close()


if __name__ == "__main__":
    main()
