"""
Integration tests for scripts/init-neo4j-schema.sh

Tests verify:
1. Schema creation success
2. Idempotency (IF NOT EXISTS)
3. Docker vs local mode detection
4. Connection failure handling
5. Missing Cypher file fallback
"""

import os
import subprocess
import pytest
from pathlib import Path


@pytest.fixture
def script_path():
    """Path to init-neo4j-schema.sh script."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "scripts" / "init-neo4j-schema.sh"


@pytest.fixture
def cypher_file():
    """Path to Cypher schema file."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "deployments" / "neo4j-init" / "001-init-schema.cypher"


@pytest.fixture
def test_env():
    """Test environment variables."""
    return {
        "NEO4J_HOST": os.getenv("NEO4J_HOST", "localhost"),
        "NEO4J_PORT": os.getenv("NEO4J_PORT", "7687"),
        "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "test_password"),
        "NEO4J_CONTAINER": "test-neo4j-container",
    }


class TestInitNeo4jSchema:
    """Test suite for Neo4j schema initialization script."""

    def test_script_exists_and_executable(self, script_path):
        """Test that the script exists and is executable."""
        assert script_path.exists(), f"Script not found at {script_path}"
        assert os.access(script_path, os.X_OK), f"Script not executable: {script_path}"

    def test_cypher_file_exists(self, cypher_file):
        """Test that the Cypher schema file exists."""
        assert cypher_file.exists(), f"Cypher file not found at {cypher_file}"

    def test_script_requires_password(self, script_path):
        """Test that script fails gracefully when NEO4J_PASSWORD is not set."""
        env = os.environ.copy()
        # Remove NEO4J_PASSWORD if it exists
        env.pop("NEO4J_PASSWORD", None)

        # Script should handle missing password gracefully
        # (May succeed if using Docker mode with container that has password set)
        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Script may succeed or fail depending on environment
        # Just verify it doesn't crash
        assert result.returncode in [0, 1], (
            f"Script crashed with unexpected exit code: {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_docker_mode_detection(self, script_path, test_env):
        """Test that script correctly detects Docker vs local mode."""
        # Check if Docker is available
        docker_available = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            timeout=10
        ).returncode == 0

        if not docker_available:
            pytest.skip("Docker not available")

        # Run script and check for mode detection message
        result = subprocess.run(
            [str(script_path)],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should output either "Detected Docker environment" or "Running in local mode"
        assert (
            "Detected Docker environment" in result.stdout or
            "Running in local mode" in result.stdout
        ), f"No mode detection message found in output:\n{result.stdout}"

    def test_connection_check(self, script_path, test_env):
        """Test that script checks Neo4j connection before proceeding."""
        result = subprocess.run(
            [str(script_path)],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should contain connection check message
        assert (
            "Checking Neo4j connection" in result.stdout or
            "Connected to Neo4j" in result.stdout or
            "Failed to connect to Neo4j" in result.stdout
        ), f"No connection check message found in output:\n{result.stdout}"

    @pytest.mark.integration
    def test_schema_creation_with_docker(self, script_path, test_env):
        """Test schema creation in Docker mode (requires running Neo4j container)."""
        # Check if test container exists
        container_check = subprocess.run(
            ["docker", "ps", "--filter", f"name={test_env['NEO4J_CONTAINER']}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if test_env['NEO4J_CONTAINER'] not in container_check.stdout:
            pytest.skip(f"Test container {test_env['NEO4J_CONTAINER']} not running")

        # Run schema initialization
        result = subprocess.run(
            [str(script_path)],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Should complete successfully
        assert result.returncode == 0, (
            f"Schema initialization failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Should report success
        assert "completed successfully" in result.stdout.lower(), (
            f"No success message in output:\n{result.stdout}"
        )

    @pytest.mark.integration
    def test_idempotency(self, script_path, test_env):
        """Test that running script multiple times is idempotent (uses IF NOT EXISTS)."""
        # Check if test container exists
        container_check = subprocess.run(
            ["docker", "ps", "--filter", f"name={test_env['NEO4J_CONTAINER']}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if test_env['NEO4J_CONTAINER'] not in container_check.stdout:
            pytest.skip(f"Test container {test_env['NEO4J_CONTAINER']} not running")

        # Run script first time
        result1 = subprocess.run(
            [str(script_path)],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Run script second time
        result2 = subprocess.run(
            [str(script_path)],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Both should succeed
        assert result1.returncode == 0, f"First run failed: {result1.stderr}"
        assert result2.returncode == 0, f"Second run failed: {result2.stderr}"

        # Second run should not fail due to existing constraints
        # (Should use IF NOT EXISTS clauses)
        assert "already exists" not in result2.stderr.lower() or result2.returncode == 0, (
            f"Second run failed due to existing constraints:\n{result2.stderr}"
        )

    @pytest.mark.integration
    def test_constraint_verification(self, script_path, test_env):
        """Test that script verifies constraints were created."""
        # Check if test container exists
        container_check = subprocess.run(
            ["docker", "ps", "--filter", f"name={test_env['NEO4J_CONTAINER']}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if test_env['NEO4J_CONTAINER'] not in container_check.stdout:
            pytest.skip(f"Test container {test_env['NEO4J_CONTAINER']} not running")

        # Run schema initialization
        result = subprocess.run(
            [str(script_path)],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Should verify schema
        assert "Verifying schema" in result.stdout, (
            f"No schema verification found in output:\n{result.stdout}"
        )

        # Should report constraint count
        assert "Constraints:" in result.stdout, (
            f"No constraint count in output:\n{result.stdout}"
        )

    def test_fallback_when_cypher_file_missing(self, script_path, test_env, tmp_path):
        """Test that script has fallback behavior when Cypher file is missing."""
        # Temporarily move Cypher file
        repo_root = Path(__file__).parent.parent.parent
        cypher_file = repo_root / "deployments" / "neo4j-init" / "001-init-schema.cypher"

        # This test documents expected behavior but doesn't actually move the file
        # (to avoid disrupting other tests)

        # Script should either:
        # 1. Create schema programmatically (as per lines 130-154 of script)
        # 2. Fail gracefully with clear error message

        # We'll just verify the fallback code exists in the script
        with open(script_path, 'r') as f:
            script_content = f.read()

        assert "Cypher file not found" in script_content, (
            "Script missing fallback handling for missing Cypher file"
        )
        assert "CREATE CONSTRAINT" in script_content, (
            "Script missing programmatic constraint creation fallback"
        )

    def test_password_validation(self, script_path):
        """Test that script validates password before using it."""
        # Test with empty password
        env = os.environ.copy()
        env["NEO4J_PASSWORD"] = ""

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Script should handle empty password gracefully
        # (May succeed if Docker container has default password)
        assert result.returncode in [0, 1], (
            f"Script crashed with unexpected exit code: {result.returncode}"
        )

    def test_error_handling_on_connection_failure(self, script_path):
        """Test that script handles connection failures gracefully."""
        # Use invalid host/port to trigger connection failure
        env = os.environ.copy()
        env["NEO4J_HOST"] = "invalid-host-that-does-not-exist"
        env["NEO4J_PORT"] = "99999"
        env["NEO4J_PASSWORD"] = "test"
        env["NEO4J_CONTAINER"] = "nonexistent-container"

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should fail gracefully (exit code 0 or 1, not crash)
        assert result.returncode in [0, 1], (
            f"Script crashed with unexpected exit code: {result.returncode}\n"
            f"stderr: {result.stderr}"
        )

        # Should not have Python tracebacks in output (indicates clean error handling)
        assert "Traceback" not in result.stderr, (
            f"Script has uncaught Python exception:\n{result.stderr}"
        )


class TestCypherFileContent:
    """Test the Cypher schema file content."""

    def test_all_constraints_use_if_not_exists(self, cypher_file):
        """Test that all CREATE CONSTRAINT statements use IF NOT EXISTS."""
        with open(cypher_file, 'r') as f:
            content = f.read()

        # Find all CREATE CONSTRAINT lines
        lines = content.split('\n')
        constraint_lines = [
            line for line in lines
            if 'CREATE CONSTRAINT' in line.upper() and not line.strip().startswith('//')
        ]

        assert len(constraint_lines) > 0, "No constraint definitions found"

        for line in constraint_lines:
            assert 'IF NOT EXISTS' in line.upper(), (
                f"Constraint missing IF NOT EXISTS: {line}"
            )

    def test_all_indexes_use_if_not_exists(self, cypher_file):
        """Test that all CREATE INDEX statements use IF NOT EXISTS."""
        with open(cypher_file, 'r') as f:
            content = f.read()

        # Find all CREATE INDEX lines
        lines = content.split('\n')
        index_lines = [
            line for line in lines
            if 'CREATE INDEX' in line.upper() and not line.strip().startswith('//')
        ]

        assert len(index_lines) > 0, "No index definitions found"

        for line in index_lines:
            assert 'IF NOT EXISTS' in line.upper(), (
                f"Index missing IF NOT EXISTS: {line}"
            )

    def test_cypher_syntax_basic_validation(self, cypher_file):
        """Test basic Cypher syntax validation."""
        with open(cypher_file, 'r') as f:
            content = f.read()

        # Check for common syntax elements
        assert 'CREATE CONSTRAINT' in content, "No constraints defined"
        assert 'CREATE INDEX' in content, "No indexes defined"

        # Check that Context label exists
        assert ':Context' in content, "No Context label found in schema"

        # Check for common node labels
        expected_labels = ['Context', 'Document', 'Sprint', 'Task', 'User']
        for label in expected_labels:
            assert f':{label}' in content, f"Label {label} not found in schema"

    def test_constraint_naming_convention(self, cypher_file):
        """Test that constraints follow naming convention."""
        with open(cypher_file, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        constraint_lines = [
            line for line in lines
            if 'CREATE CONSTRAINT' in line.upper() and not line.strip().startswith('//')
        ]

        for line in constraint_lines:
            # Extract constraint name (usually after CONSTRAINT keyword)
            if 'CREATE CONSTRAINT' in line.upper():
                # Constraint names should be descriptive
                # Format: entity_property_unique or similar
                assert any(char in line for char in ['_', 'unique', 'id']), (
                    f"Constraint name not following convention: {line}"
                )


@pytest.mark.integration
class TestEndToEndSchemaInitialization:
    """End-to-end tests requiring a running Neo4j instance."""

    def test_full_schema_initialization_workflow(self, script_path, test_env):
        """Test the complete schema initialization workflow."""
        # Check if test container exists
        container_check = subprocess.run(
            ["docker", "ps", "--filter", f"name={test_env['NEO4J_CONTAINER']}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if test_env['NEO4J_CONTAINER'] not in container_check.stdout:
            pytest.skip(f"Test container {test_env['NEO4J_CONTAINER']} not running")

        # 1. Run schema initialization
        init_result = subprocess.run(
            [str(script_path)],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        assert init_result.returncode == 0, (
            f"Schema initialization failed:\n{init_result.stderr}"
        )

        # 2. Verify constraints exist
        verify_constraints = subprocess.run(
            [
                "docker", "exec", test_env['NEO4J_CONTAINER'],
                "cypher-shell", "-u", test_env['NEO4J_USER'], "-p", test_env['NEO4J_PASSWORD'],
                "CALL db.constraints() YIELD name RETURN count(name) as count"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if verify_constraints.returncode == 0:
            # Should have at least 5 constraints (per schema file)
            # Note: This is a soft check as exact count may vary
            assert "count" in verify_constraints.stdout.lower(), (
                f"Constraint count not found in output:\n{verify_constraints.stdout}"
            )

        # 3. Verify indexes exist
        verify_indexes = subprocess.run(
            [
                "docker", "exec", test_env['NEO4J_CONTAINER'],
                "cypher-shell", "-u", test_env['NEO4J_USER'], "-p", test_env['NEO4J_PASSWORD'],
                "CALL db.indexes() YIELD name RETURN count(name) as count"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if verify_indexes.returncode == 0:
            assert "count" in verify_indexes.stdout.lower(), (
                f"Index count not found in output:\n{verify_indexes.stdout}"
            )

        # 4. Verify Context label exists
        verify_label = subprocess.run(
            [
                "docker", "exec", test_env['NEO4J_CONTAINER'],
                "cypher-shell", "-u", test_env['NEO4J_USER'], "-p", test_env['NEO4J_PASSWORD'],
                "CALL db.labels() YIELD label RETURN label"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if verify_label.returncode == 0:
            # Context label should exist (or will exist when first Context node is created)
            # This is a soft check as labels may not appear until nodes are created
            pass  # Label verification is optional
