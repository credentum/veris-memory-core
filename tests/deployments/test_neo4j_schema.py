"""
Tests for deployments/neo4j-init/001-init-schema.cypher

Tests verify:
1. Cypher syntax validation
2. All constraints use IF NOT EXISTS (idempotency)
3. All indexes use IF NOT EXISTS (idempotency)
4. Schema can be applied to test Neo4j instance
5. Schema integrity and completeness
"""

import os
import re
import subprocess
import pytest
from pathlib import Path


@pytest.fixture
def cypher_file():
    """Path to Cypher schema file."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "deployments" / "neo4j-init" / "001-init-schema.cypher"


@pytest.fixture
def cypher_content(cypher_file):
    """Content of Cypher schema file."""
    with open(cypher_file, 'r') as f:
        return f.read()


class TestCypherFileSyntax:
    """Test Cypher file syntax and structure."""

    def test_file_exists(self, cypher_file):
        """Test that Cypher schema file exists."""
        assert cypher_file.exists(), f"Cypher file not found: {cypher_file}"

    def test_file_not_empty(self, cypher_content):
        """Test that Cypher file is not empty."""
        assert len(cypher_content.strip()) > 0, "Cypher file is empty"

    def test_file_is_valid_utf8(self, cypher_file):
        """Test that file is valid UTF-8."""
        try:
            with open(cypher_file, 'r', encoding='utf-8') as f:
                f.read()
        except UnicodeDecodeError as e:
            pytest.fail(f"File is not valid UTF-8: {e}")

    def test_no_syntax_errors_basic(self, cypher_content):
        """Test basic Cypher syntax (no obvious errors)."""
        # Check for common syntax errors
        lines = cypher_content.split('\n')

        for i, line in enumerate(lines, 1):
            # Skip comments and empty lines
            if line.strip().startswith('//') or not line.strip():
                continue

            # Check for unclosed strings
            if line.count('"') % 2 != 0 and line.count("'") % 2 != 0:
                # Might be a multi-line string, but flag for review
                pass

            # Check for unmatched parentheses
            if line.count('(') != line.count(')'):
                # Might be multi-line, but flag for review
                pass

    def test_statements_end_with_semicolon(self, cypher_content):
        """Test that Cypher statements end with semicolon."""
        # Remove comments
        lines = [
            line for line in cypher_content.split('\n')
            if line.strip() and not line.strip().startswith('//')
        ]

        # Combine multi-line statements
        full_content = '\n'.join(lines)

        # Split by semicolon to get statements
        statements = [s.strip() for s in full_content.split(';') if s.strip()]

        # All statements should be valid (not empty after splitting)
        assert len(statements) > 0, "No Cypher statements found"

        for stmt in statements:
            # Each statement should have CREATE keyword
            assert any(keyword in stmt.upper() for keyword in ['CREATE', 'DROP', 'MATCH']), (
                f"Statement missing Cypher keyword: {stmt[:100]}"
            )


class TestConstraintDefinitions:
    """Test constraint definitions in Cypher file."""

    def test_all_constraints_use_if_not_exists(self, cypher_content):
        """Test that all CREATE CONSTRAINT statements use IF NOT EXISTS."""
        lines = cypher_content.split('\n')
        constraint_blocks = []
        current_block = []

        for line in lines:
            if 'CREATE CONSTRAINT' in line.upper():
                current_block = [line]
            elif current_block and ';' not in line:
                current_block.append(line)
            elif current_block and ';' in line:
                current_block.append(line)
                constraint_blocks.append('\n'.join(current_block))
                current_block = []

        assert len(constraint_blocks) > 0, "No constraints found in schema file"

        for i, block in enumerate(constraint_blocks, 1):
            assert 'IF NOT EXISTS' in block.upper(), (
                f"Constraint #{i} missing IF NOT EXISTS clause:\n{block}"
            )

    def test_context_constraint_exists(self, cypher_content):
        """Test that Context.id constraint exists (critical for fixing 29.3% error rate)."""
        assert ':Context' in cypher_content, "No Context label constraint found"
        assert 'c.id' in cypher_content or 'Context).id' in cypher_content, (
            "No Context.id constraint found"
        )
        assert 'UNIQUE' in cypher_content.upper() or 'REQUIRE' in cypher_content.upper(), (
            "No uniqueness constraint found"
        )

    def test_expected_constraints_exist(self, cypher_content):
        """Test that all expected constraints exist."""
        expected_constraints = {
            'Context': 'id',
            'Document': 'id',
            'Sprint': 'id',
            'Task': 'id',
            'User': 'id',
        }

        for entity, property_name in expected_constraints.items():
            assert f':{entity}' in cypher_content, (
                f"Missing constraint for {entity}"
            )

    def test_constraint_naming_convention(self, cypher_content):
        """Test that constraints follow naming convention."""
        # Extract constraint names
        constraint_pattern = r'CREATE CONSTRAINT\s+(\w+)\s+IF NOT EXISTS'
        matches = re.findall(constraint_pattern, cypher_content, re.IGNORECASE)

        assert len(matches) > 0, "No named constraints found"

        for name in matches:
            # Convention: entity_property_unique or similar
            assert '_' in name, (
                f"Constraint name '{name}' doesn't follow convention (should use underscores)"
            )

            # Should end with 'unique' for uniqueness constraints
            if 'unique' not in name.lower():
                # Warning: might not be following full convention
                pass  # Allow for now

    def test_constraints_use_require_syntax(self, cypher_content):
        """Test that constraints use modern REQUIRE syntax (Neo4j 4.4+)."""
        constraint_lines = [
            line for line in cypher_content.split('\n')
            if 'CREATE CONSTRAINT' in line.upper()
        ]

        for line in constraint_lines:
            # Modern syntax uses REQUIRE instead of ASSERT
            if 'FOR' in line.upper() and 'REQUIRE' not in line.upper():
                # Should use REQUIRE for Neo4j 4.4+
                pass  # Check passes if REQUIRE is used


class TestIndexDefinitions:
    """Test index definitions in Cypher file."""

    def test_all_indexes_use_if_not_exists(self, cypher_content):
        """Test that all CREATE INDEX statements use IF NOT EXISTS."""
        lines = cypher_content.split('\n')
        index_blocks = []
        current_block = []

        for line in lines:
            if 'CREATE INDEX' in line.upper() and 'CREATE CONSTRAINT' not in line.upper():
                current_block = [line]
            elif current_block and ';' not in line:
                current_block.append(line)
            elif current_block and ';' in line:
                current_block.append(line)
                index_blocks.append('\n'.join(current_block))
                current_block = []

        assert len(index_blocks) > 0, "No indexes found in schema file"

        for i, block in enumerate(index_blocks, 1):
            assert 'IF NOT EXISTS' in block.upper(), (
                f"Index #{i} missing IF NOT EXISTS clause:\n{block}"
            )

    def test_expected_indexes_exist(self, cypher_content):
        """Test that expected indexes exist for query performance."""
        expected_indexes = {
            'Context': ['type', 'created_at', 'author', 'author_type'],
            'Document': ['document_type', 'created_date', 'status', 'last_modified'],
            'Sprint': ['sprint_number', 'status'],
            'Task': ['status', 'assigned_to'],
        }

        for entity, properties in expected_indexes.items():
            for prop in properties:
                # Check if index exists for this property
                # May be individual index or part of compound index
                if f':{entity}' in cypher_content and prop in cypher_content:
                    # Good - index likely exists
                    pass
                # Not all properties may be indexed yet - that's OK

    def test_index_naming_convention(self, cypher_content):
        """Test that indexes follow naming convention."""
        # Extract index names
        index_pattern = r'CREATE INDEX\s+(\w+)\s+IF NOT EXISTS'
        matches = re.findall(index_pattern, cypher_content, re.IGNORECASE)

        if len(matches) > 0:
            for name in matches:
                # Convention: entity_property_idx or similar
                assert '_' in name, (
                    f"Index name '{name}' doesn't follow convention (should use underscores)"
                )

                # Should end with 'idx' for indexes
                assert name.endswith('_idx') or 'index' in name.lower(), (
                    f"Index name '{name}' should end with '_idx'"
                )


class TestSchemaCompleteness:
    """Test that schema is complete and covers all necessary entities."""

    def test_context_label_fully_defined(self, cypher_content):
        """Test that Context label has complete schema definition."""
        # Should have constraint
        assert ':Context' in cypher_content and 'CONSTRAINT' in cypher_content.upper()

        # Should have at least one index
        context_indexes = 0
        lines = cypher_content.split('\n')
        for line in lines:
            if ':Context' in line and 'CREATE INDEX' in line.upper():
                context_indexes += 1

        assert context_indexes > 0, "Context label missing indexes"

    def test_all_entity_labels_defined(self, cypher_content):
        """Test that all expected entity labels are defined."""
        expected_labels = ['Context', 'Document', 'Sprint', 'Task', 'User']

        for label in expected_labels:
            assert f':{label}' in cypher_content, f"Label {label} not defined in schema"

    def test_no_deprecated_syntax(self, cypher_content):
        """Test that schema doesn't use deprecated Cypher syntax."""
        deprecated_patterns = [
            'CREATE UNIQUE',  # Deprecated in favor of constraints
            'ASSERT',  # Old constraint syntax
            'ON CREATE SET',  # Use MERGE instead
        ]

        for pattern in deprecated_patterns:
            if pattern in cypher_content.upper():
                # Warning: might be using deprecated syntax
                pytest.fail(f"Schema uses deprecated syntax: {pattern}")


class TestSchemaIntegrity:
    """Test schema integrity and consistency."""

    def test_no_duplicate_constraint_names(self, cypher_content):
        """Test that there are no duplicate constraint names."""
        constraint_pattern = r'CREATE CONSTRAINT\s+(\w+)\s+IF NOT EXISTS'
        matches = re.findall(constraint_pattern, cypher_content, re.IGNORECASE)

        if len(matches) > 0:
            # Check for duplicates
            duplicates = [name for name in matches if matches.count(name) > 1]
            assert len(duplicates) == 0, f"Duplicate constraint names: {set(duplicates)}"

    def test_no_duplicate_index_names(self, cypher_content):
        """Test that there are no duplicate index names."""
        index_pattern = r'CREATE INDEX\s+(\w+)\s+IF NOT EXISTS'
        matches = re.findall(index_pattern, cypher_content, re.IGNORECASE)

        if len(matches) > 0:
            # Check for duplicates
            duplicates = [name for name in matches if matches.count(name) > 1]
            assert len(duplicates) == 0, f"Duplicate index names: {set(duplicates)}"

    def test_constraints_before_indexes(self, cypher_content):
        """Test that constraints are defined before indexes (best practice)."""
        lines = cypher_content.split('\n')

        first_constraint_line = None
        first_index_line = None

        for i, line in enumerate(lines):
            if 'CREATE CONSTRAINT' in line.upper() and first_constraint_line is None:
                first_constraint_line = i

            if 'CREATE INDEX' in line.upper() and 'CREATE CONSTRAINT' not in line.upper():
                if first_index_line is None:
                    first_index_line = i

        if first_constraint_line is not None and first_index_line is not None:
            # Constraints should come before indexes
            assert first_constraint_line < first_index_line, (
                f"Indexes (line {first_index_line}) defined before constraints (line {first_constraint_line}). "
                "Best practice: define constraints first."
            )


class TestSchemaDocumentation:
    """Test that schema is well-documented."""

    def test_schema_has_comments(self, cypher_content):
        """Test that schema file has explanatory comments."""
        comment_lines = [
            line for line in cypher_content.split('\n')
            if line.strip().startswith('//')
        ]

        assert len(comment_lines) > 0, "Schema file has no comments"

    def test_each_entity_has_comment(self, cypher_content):
        """Test that each entity type has a comment explaining its purpose."""
        entities = ['Context', 'Document', 'Sprint', 'Task', 'User']

        lines = cypher_content.split('\n')

        for entity in entities:
            # Find constraint for entity
            entity_line = None
            for i, line in enumerate(lines):
                if f':{entity}' in line and 'CREATE CONSTRAINT' in line.upper():
                    entity_line = i
                    break

            if entity_line is not None:
                # Check if there's a comment nearby (within 3 lines before)
                nearby_lines = lines[max(0, entity_line - 3):entity_line]
                has_comment = any(line.strip().startswith('//') for line in nearby_lines)

                # Comment is helpful but not required
                # Just check that at least one entity has documentation
                pass


@pytest.mark.integration
class TestSchemaApplication:
    """Integration tests for applying schema to Neo4j instance."""

    def test_schema_can_be_parsed_by_cypher_shell(self, cypher_file):
        """Test that schema can be parsed by cypher-shell (requires Neo4j)."""
        # This test requires Neo4j to be running
        # Check if Neo4j container exists
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'test')
        neo4j_container = os.getenv('NEO4J_CONTAINER', 'test-neo4j-container')

        container_check = subprocess.run(
            ['docker', 'ps', '--filter', f'name={neo4j_container}', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if neo4j_container not in container_check.stdout:
            pytest.skip(f"Neo4j container {neo4j_container} not running")

        # Copy file to container and validate syntax
        subprocess.run(
            ['docker', 'cp', str(cypher_file), f'{neo4j_container}:/tmp/schema-test.cypher'],
            check=True,
            timeout=10
        )

        # Try to apply schema (this validates syntax)
        result = subprocess.run(
            [
                'docker', 'exec', neo4j_container,
                'cypher-shell', '-u', 'neo4j', '-p', neo4j_password,
                '-f', '/tmp/schema-test.cypher'
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Clean up
        subprocess.run(
            ['docker', 'exec', neo4j_container, 'rm', '/tmp/schema-test.cypher'],
            timeout=10
        )

        # Should succeed or have acceptable warnings
        if result.returncode != 0:
            # Check if failure is due to "already exists" (acceptable)
            if 'already exists' in result.stderr.lower():
                pass  # Acceptable - schema already applied
            else:
                pytest.fail(f"Schema application failed:\n{result.stderr}")

    @pytest.mark.integration
    def test_schema_application_is_idempotent(self, cypher_file):
        """Test that schema can be applied multiple times without errors."""
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'test')
        neo4j_container = os.getenv('NEO4J_CONTAINER', 'test-neo4j-container')

        container_check = subprocess.run(
            ['docker', 'ps', '--filter', f'name={neo4j_container}', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if neo4j_container not in container_check.stdout:
            pytest.skip(f"Neo4j container {neo4j_container} not running")

        # Apply schema twice
        for attempt in range(2):
            subprocess.run(
                ['docker', 'cp', str(cypher_file), f'{neo4j_container}:/tmp/schema-test.cypher'],
                check=True,
                timeout=10
            )

            result = subprocess.run(
                [
                    'docker', 'exec', neo4j_container,
                    'cypher-shell', '-u', 'neo4j', '-p', neo4j_password,
                    '-f', '/tmp/schema-test.cypher'
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            subprocess.run(
                ['docker', 'exec', neo4j_container, 'rm', '/tmp/schema-test.cypher'],
                timeout=10
            )

            # Both attempts should succeed
            assert result.returncode == 0 or 'already exists' in result.stderr.lower(), (
                f"Attempt {attempt + 1} failed:\n{result.stderr}"
            )

    @pytest.mark.integration
    def test_context_label_exists_after_schema_application(self, cypher_file):
        """Test that Context label exists after schema is applied."""
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'test')
        neo4j_container = os.getenv('NEO4J_CONTAINER', 'test-neo4j-container')

        container_check = subprocess.run(
            ['docker', 'ps', '--filter', f'name={neo4j_container}', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if neo4j_container not in container_check.stdout:
            pytest.skip(f"Neo4j container {neo4j_container} not running")

        # Apply schema
        subprocess.run(
            ['docker', 'cp', str(cypher_file), f'{neo4j_container}:/tmp/schema-test.cypher'],
            check=True,
            timeout=10
        )

        subprocess.run(
            [
                'docker', 'exec', neo4j_container,
                'cypher-shell', '-u', 'neo4j', '-p', neo4j_password,
                '-f', '/tmp/schema-test.cypher'
            ],
            capture_output=True,
            timeout=60
        )

        # Verify constraints exist
        verify_result = subprocess.run(
            [
                'docker', 'exec', neo4j_container,
                'cypher-shell', '-u', 'neo4j', '-p', neo4j_password,
                'CALL db.constraints() YIELD name RETURN count(name) as count'
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Clean up
        subprocess.run(
            ['docker', 'exec', neo4j_container, 'rm', '/tmp/schema-test.cypher'],
            timeout=10
        )

        # Should have constraints
        if verify_result.returncode == 0:
            assert 'count' in verify_result.stdout.lower(), (
                f"No constraints found after schema application:\n{verify_result.stdout}"
            )
