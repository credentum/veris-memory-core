#!/usr/bin/env python3
"""
Comprehensive test suite for query_validator.py module.
Tests all validation methods, Cypher query validation, and security features.
"""

import pytest

from src.core.query_validator import (  # noqa: E402
    CypherOperation,
    QueryValidator,
    validate_cypher_query,
)


class TestCypherOperation:
    """Test the CypherOperation enum."""

    def test_read_operations(self):
        """Test that read operations are correctly defined."""
        read_ops = {
            CypherOperation.MATCH,
            CypherOperation.WITH,
            CypherOperation.RETURN,
            CypherOperation.WHERE,
            CypherOperation.ORDER_BY,
            CypherOperation.LIMIT,
            CypherOperation.SKIP,
            CypherOperation.UNWIND,
            CypherOperation.OPTIONAL_MATCH,
            CypherOperation.CALL,
        }

        assert read_ops == QueryValidator.ALLOWED_OPERATIONS

    def test_write_operations(self):
        """Test that write operations are correctly defined."""
        write_ops = {
            CypherOperation.CREATE,
            CypherOperation.DELETE,
            CypherOperation.DETACH_DELETE,
            CypherOperation.SET,
            CypherOperation.REMOVE,
            CypherOperation.MERGE,
            CypherOperation.DROP,
            CypherOperation.FOREACH,
        }

        assert write_ops == QueryValidator.FORBIDDEN_OPERATIONS

    def test_enum_values(self):
        """Test that enum values match expected strings."""
        assert CypherOperation.MATCH.value == "MATCH"
        assert CypherOperation.CREATE.value == "CREATE"
        assert CypherOperation.DELETE.value == "DELETE"
        assert CypherOperation.SET.value == "SET"


class TestQueryValidatorBasics:
    """Test basic QueryValidator functionality."""

    def test_empty_query_validation(self):
        """Test validation of empty queries."""
        is_valid, error = QueryValidator.validate_query("")
        assert not is_valid
        assert error == "Query cannot be empty"

        is_valid, error = QueryValidator.validate_query("   ")
        assert not is_valid
        assert error == "Query cannot be empty"

        is_valid, error = QueryValidator.validate_query(None)
        assert not is_valid
        assert error == "Query cannot be empty"

    def test_query_length_validation(self):
        """Test validation of query length limits."""
        # Test maximum length query
        long_query = "MATCH (n) RETURN n " + "a" * 10000
        is_valid, error = QueryValidator.validate_query(long_query)
        assert not is_valid
        assert error == "Query exceeds maximum length"

        # Test query just under limit
        normal_query = "MATCH (n) RETURN n " + "a" * 100
        is_valid, error = QueryValidator.validate_query(normal_query)
        assert is_valid
        assert error is None


class TestValidQueries:
    """Test validation of valid Cypher queries."""

    def test_simple_match_query(self):
        """Test simple MATCH queries."""
        queries = [
            "MATCH (n) RETURN n",
            "MATCH (n:Person) RETURN n.name",
            "MATCH (a)-[r]->(b) RETURN a, r, b",
            "MATCH (n:Person {name: 'John'}) RETURN n",
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert is_valid, f"Query '{query}' should be valid, but got error: {error}"
            assert error is None

    def test_complex_read_queries(self):
        """Test complex read-only queries."""
        queries = [
            "MATCH (n:Person) WHERE n.age > 25 RETURN n.name ORDER BY n.name LIMIT 10",
            "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.name = 'Alice' RETURN b.name",
            "OPTIONAL MATCH (n:Person)-[:LIVES_IN]->(c:City) RETURN n.name, c.name",
            "MATCH (n) WITH n MATCH (n)-[r]->(m) RETURN n, r, m",
            "UNWIND [1, 2, 3] AS x RETURN x",
            "CALL db.labels() YIELD label RETURN label",
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert is_valid, f"Query '{query}' should be valid, but got error: {error}"
            assert error is None

    def test_queries_with_parameters(self):
        """Test queries that would use parameters."""
        queries = [
            "MATCH (n:Person) WHERE n.name = $name RETURN n",
            "MATCH (n) WHERE n.id = $nodeId RETURN n",
            "MATCH (n:Person) WHERE n.age > $minAge RETURN n ORDER BY n.age LIMIT $limit",
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert is_valid, f"Query '{query}' should be valid, but got error: {error}"
            assert error is None


class TestForbiddenOperations:
    """Test detection of forbidden write operations."""

    def test_create_operations(self):
        """Test detection of CREATE operations."""
        queries = [
            "CREATE (n:Person {name: 'John'})",
            "MATCH (n) CREATE (n)-[:KNOWS]->(m:Person {name: 'Jane'})",
            "create (n:Test)",  # Case insensitive
            "CREATE INDEX ON :Person(name)",
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            assert "CREATE" in error

    def test_delete_operations(self):
        """Test detection of DELETE operations."""
        queries = [
            "MATCH (n:Person) DELETE n",
            "MATCH (n:Person)-[r]->(m) DELETE r",
            "MATCH (n) DETACH DELETE n",
            "delete n",  # Case insensitive
            "DETACH delete n",  # Case insensitive
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            assert "DELETE" in error

    def test_set_operations(self):
        """Test detection of SET operations."""
        queries = [
            "MATCH (n:Person) SET n.name = 'NewName'",
            "MATCH (n) SET n:NewLabel",
            "MATCH (n) SET n += {property: 'value'}",
            "set n.name = 'test'",  # Case insensitive
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            assert "SET" in error

    def test_remove_operations(self):
        """Test detection of REMOVE operations."""
        queries = [
            "MATCH (n:Person) REMOVE n.name",
            "MATCH (n) REMOVE n:Label",
            "remove n.property",  # Case insensitive
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            assert "REMOVE" in error

    def test_merge_operations(self):
        """Test detection of MERGE operations."""
        queries = [
            "MERGE (n:Person {name: 'John'})",
            "MATCH (a) MERGE (a)-[:KNOWS]->(b:Person {name: 'Jane'})",
            "merge (n:Test)",  # Case insensitive
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            assert "MERGE" in error

    def test_drop_operations(self):
        """Test detection of DROP operations."""
        queries = [
            "DROP INDEX ON :Person(name)",
            "DROP CONSTRAINT ON (n:Person) ASSERT n.email IS UNIQUE",
            "drop index person_name",  # Case insensitive
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            assert "DROP" in error

    def test_foreach_operations(self):
        """Test detection of FOREACH operations."""
        queries = [
            "FOREACH (x IN [1, 2, 3] | CREATE (n:Number {value: x}))",
            "MATCH (n) FOREACH (r IN n.relationships | DELETE r)",
            "foreach (item in list | create (n:Item {name: item}))",  # Case insensitive
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            # These queries will be caught by CREATE/DELETE detection first
            assert any(
                op in error for op in ["FOREACH", "CREATE", "DELETE"]
            ), f"Expected operation error, got: {error}"


class TestInjectionDetection:
    """Test detection of potential injection attacks."""

    def test_command_chaining_detection(self):
        """Test detection of command chaining with semicolons."""
        queries = [
            "MATCH (n) RETURN n; CREATE (m:Evil)",
            "MATCH (n) RETURN n;CREATE (x)",
            "MATCH (n) RETURN n ; DROP DATABASE",
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            # These queries will be caught by forbidden operation detection first
            assert any(
                phrase in error for phrase in ["Command chaining is not allowed", "CREATE", "DROP"]
            ), f"Expected command chaining or operation error, got: {error}"

    def test_pure_command_chaining_without_forbidden_ops(self):
        """Test command chaining without forbidden operations to trigger specific error path."""
        # This should trigger command chaining error without hitting forbidden operations first
        query = "MATCH (n) RETURN n; MATCH (m) RETURN m"
        is_valid, error = QueryValidator.validate_query(query)
        assert not is_valid, f"Query '{query}' should be invalid"
        assert "Command chaining is not allowed" in error

    def test_comment_injection_detection(self):
        """Test detection of comment-based injection."""
        queries = [
            "MATCH (n) /* CREATE (x) */ RETURN n",
            "MATCH (n) RETURN n -- CREATE (x)",
            "MATCH (n) RETURN n /* hidden command CREATE (evil) */",
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            # These will be caught by forbidden operations first
            assert any(
                phrase in error for phrase in ["Potential injection pattern detected", "CREATE"]
            ), f"Expected injection or operation error, got: {error}"

    def test_escape_sequence_detection(self):
        """Test detection of escape sequences."""
        queries = [
            "MATCH (n) WHERE n.name = '\\x41\\x42'",  # Hex escapes
            "MATCH (n) WHERE n.name = '\\101\\102'",  # Octal escapes
            "MATCH (n) WHERE n.name = char(65)",  # Character construction
            "MATCH (n) WHERE n.name = concat('A', 'B')",  # String concatenation
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            assert "Potential injection pattern detected" in error


class TestStringLiteralHandling:
    """Test string literal removal and handling."""

    def test_remove_string_literals(self):
        """Test removal of string literals."""
        test_cases = [
            ("MATCH (n {name: 'John'}) RETURN n", "MATCH (n {name: ''}) RETURN n"),
            ('MATCH (n {name: "John"}) RETURN n', 'MATCH (n {name: ""}) RETURN n'),
            ("MATCH (n:`Person Name`) RETURN n", "MATCH (n:``) RETURN n"),
            (
                "MATCH (n {name: 'John', title: 'Dr.'}) RETURN n",
                "MATCH (n {name: '', title: ''}) RETURN n",
            ),
        ]

        for original, expected in test_cases:
            result = QueryValidator._remove_string_literals(original)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_string_literals_prevent_false_positives(self):
        """Test that string literals don't trigger false positives."""
        queries = [
            "MATCH (n {description: 'CREATE a new record'}) RETURN n",
            "MATCH (n {text: 'DELETE old data'}) RETURN n",
            "MATCH (n {command: 'SET property value'}) RETURN n",
            'MATCH (n {note: "MERGE this data"}) RETURN n',
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert is_valid, f"Query '{query}' should be valid, but got error: {error}"
            assert error is None


class TestSyntaxValidation:
    """Test syntax validation features."""

    def test_unbalanced_parentheses_in_query(self):
        """Test that unbalanced parentheses are caught in full query validation."""
        query = "MATCH (n RETURN n"  # Missing closing parenthesis
        is_valid, error = QueryValidator.validate_query(query)
        assert not is_valid
        assert "Unbalanced parentheses detected" in error

    def test_unbalanced_quotes_in_query(self):
        """Test that unbalanced quotes are caught in full query validation."""
        query = "MATCH (n {name: 'John}) RETURN n"  # Missing closing quote
        is_valid, error = QueryValidator.validate_query(query)
        assert not is_valid
        assert "Unbalanced quotes detected" in error

    def test_parentheses_validation(self):
        """Test parentheses balance validation."""
        valid_queries = [
            "MATCH (n) RETURN n",
            "MATCH (a)-[r]->(b) RETURN a, r, b",
            "MATCH (n {name: 'test'}) RETURN n",
            "MATCH (n) WHERE n.age > (20 + 5) RETURN n",
        ]

        invalid_queries = [
            "MATCH (n RETURN n",  # Missing closing parenthesis
            "MATCH n) RETURN n",  # Missing opening parenthesis
            "MATCH ((n) RETURN n",  # Extra opening parenthesis
            "MATCH (n)) RETURN n",  # Extra closing parenthesis
        ]

        for query in valid_queries:
            assert QueryValidator._validate_parentheses(
                query
            ), f"Query '{query}' should have balanced parentheses"

        for query in invalid_queries:
            assert not QueryValidator._validate_parentheses(
                query
            ), f"Query '{query}' should have unbalanced parentheses"

    def test_quotes_validation(self):
        """Test quotes balance validation."""
        valid_queries = [
            "MATCH (n {name: 'John'}) RETURN n",
            'MATCH (n {name: "John"}) RETURN n',
            "MATCH (n:`Label Name`) RETURN n",
            "MATCH (n {name: 'John', title: 'Dr.'}) RETURN n",
        ]

        invalid_queries = [
            "MATCH (n {name: 'John}) RETURN n",  # Missing closing single quote
            'MATCH (n {name: "John}) RETURN n',  # Missing closing double quote
            "MATCH (n:`Label Name) RETURN n",  # Missing closing backtick
            "MATCH (n {name: 'John', title: Dr.'}) RETURN n",  # Missing opening single quote
        ]

        for query in valid_queries:
            assert QueryValidator._validate_quotes(
                query
            ), f"Query '{query}' should have balanced quotes"

        for query in invalid_queries:
            assert not QueryValidator._validate_quotes(
                query
            ), f"Query '{query}' should have unbalanced quotes"

    def test_escaped_quotes_handling(self):
        """Test handling of escaped quotes."""
        queries_with_escaped_quotes = [
            "MATCH (n {name: 'John\\'s'}) RETURN n",  # Escaped single quote
            'MATCH (n {name: "John\\"s"}) RETURN n',  # Escaped double quote
        ]

        for query in queries_with_escaped_quotes:
            assert QueryValidator._validate_quotes(
                query
            ), f"Query '{query}' should handle escaped quotes correctly"


class TestParameterSanitization:
    """Test parameter sanitization functionality."""

    def test_valid_parameter_names(self):
        """Test validation of parameter names."""
        valid_params = {
            "name": "John",
            "age": 25,
            "is_active": True,
            "_private": "value",
            "param_123": "test",
        }

        result = QueryValidator.sanitize_parameters(valid_params)
        assert result == valid_params

    def test_invalid_parameter_names(self):
        """Test rejection of invalid parameter names."""
        invalid_params_list = [
            {"123invalid": "value"},  # Starts with number
            {"param-name": "value"},  # Contains hyphen
            {"param name": "value"},  # Contains space
            {"param.name": "value"},  # Contains dot
            {"": "value"},  # Empty name
        ]

        for invalid_params in invalid_params_list:
            with pytest.raises(ValueError, match="Invalid parameter name"):
                QueryValidator.sanitize_parameters(invalid_params)

    def test_string_parameter_sanitization(self):
        """Test sanitization of string parameters."""
        params = {
            "text": "normal text",
            "with_control": "text\x00with\x1fcontrol\x7fchars",
            "clean": "already clean text",
        }

        result = QueryValidator.sanitize_parameters(params)

        assert result["text"] == "normal text"
        assert result["with_control"] == "textwithcontrolchars"  # Control chars removed
        assert result["clean"] == "already clean text"

    def test_non_string_parameters(self):
        """Test handling of non-string parameters."""
        params = {
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none_value": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
        }

        result = QueryValidator.sanitize_parameters(params)
        assert result == params  # Non-strings should pass through unchanged


class TestSafeQueryCreation:
    """Test safe query creation functionality."""

    def test_basic_safe_query(self):
        """Test creation of basic safe queries."""
        query = QueryValidator.create_safe_query("(n:Person)")
        expected = "MATCH (n:Person) RETURN *"
        assert query == expected

    def test_safe_query_with_return_fields(self):
        """Test safe query creation with specific return fields."""
        query = QueryValidator.create_safe_query(
            pattern="(n:Person)", return_fields=["n.name", "n.age"]
        )
        expected = "MATCH (n:Person) RETURN n.name, n.age"
        assert query == expected

    def test_safe_query_with_filters(self):
        """Test safe query creation with WHERE filters."""
        query = QueryValidator.create_safe_query(
            pattern="(n:Person)", filters={"n.age": 25, "n.name": "John"}
        )
        # Note: Order of WHERE conditions may vary
        assert "MATCH (n:Person)" in query
        assert "WHERE" in query
        assert "n.age = 25" in query
        assert "n.name = 'John'" in query
        assert "RETURN *" in query

    def test_safe_query_with_limit(self):
        """Test safe query creation with LIMIT."""
        query = QueryValidator.create_safe_query(pattern="(n:Person)", limit=10)
        expected = "MATCH (n:Person) RETURN * LIMIT 10"
        assert query == expected

    def test_safe_query_complete_example(self):
        """Test safe query creation with all parameters."""
        query = QueryValidator.create_safe_query(
            pattern="(n:Person)-[:KNOWS]->(m:Person)",
            filters={"n.name": "Alice", "m.age": 30},
            return_fields=["n.name", "m.name"],
            limit=5,
        )

        assert "MATCH (n:Person)-[:KNOWS]->(m:Person)" in query
        assert "WHERE" in query
        assert "n.name = 'Alice'" in query
        assert "m.age = 30" in query
        assert "RETURN n.name, m.name" in query
        assert "LIMIT 5" in query

    def test_safe_query_filter_types(self):
        """Test safe query creation with different filter value types."""
        query = QueryValidator.create_safe_query(
            pattern="(n:Person)",
            filters={
                "n.name": "John",
                "n.age": 25,
                "n.salary": 50000.50,
                "n.active": True,
                "n.inactive": False,
                "n.description": None,
            },
        )

        assert "n.name = 'John'" in query
        assert "n.age = 25" in query
        assert "n.salary = 50000.5" in query
        assert "n.active = true" in query  # Correctly lowercase for Cypher
        assert "n.inactive = false" in query
        assert "n.description IS NULL" in query

    def test_boolean_conversion_edge_case(self):
        """Test boolean conversion specifically to ensure code path is covered."""
        # Test individual boolean values to try to trigger the lowercase conversion
        query_true = QueryValidator.create_safe_query(
            pattern="(n:Person)", filters={"n.is_active": True}
        )
        query_false = QueryValidator.create_safe_query(
            pattern="(n:Person)", filters={"n.is_deleted": False}
        )

        # The implementation should convert booleans - check both cases
        assert ("n.is_active = true" in query_true) or ("n.is_active = True" in query_true)
        assert ("n.is_deleted = false" in query_false) or ("n.is_deleted = False" in query_false)

    def test_safe_query_invalid_inputs(self):
        """Test safe query creation with invalid inputs."""
        # Invalid filter key
        with pytest.raises(ValueError, match="Invalid filter key"):
            QueryValidator.create_safe_query(pattern="(n:Person)", filters={"n-invalid": "value"})

        # Invalid return field
        with pytest.raises(ValueError, match="Invalid return field"):
            QueryValidator.create_safe_query(
                pattern="(n:Person)", return_fields=["n.name", "invalid-field"]
            )

        # Invalid limit - negative
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            QueryValidator.create_safe_query(pattern="(n:Person)", limit=-1)

        # Test non-integer limit
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            QueryValidator.create_safe_query(pattern="(n:Person)", limit="10")

        # Note: limit=0 is treated as falsy and no LIMIT clause is added,
        # but it doesn't raise an error in the current implementation
        # This could be considered a bug in the original implementation


class TestStringEscaping:
    """Test string escaping functionality."""

    def test_escape_string(self):
        """Test string escaping for safe inclusion in queries."""
        test_cases = [
            ("normal text", "normal text"),
            ("text with 'single quotes'", "text with \\\\'single quotes\\\\'"),
            ("text with\\backslashes", "text with\\\\backslashes"),
            ("text\x00with\x1fcontrol\x7fchars", "textwithcontrolchars"),
            ("mixed: 'quotes' and\\slashes", "mixed: \\\\'quotes\\\\' and\\\\slashes"),
        ]

        for input_str, expected in test_cases:
            result = QueryValidator._escape_string(input_str)
            assert result == expected, f"Expected '{expected}', got '{result}'"


class TestCommandChaining:
    """Test command chaining detection."""

    def test_has_command_chaining(self):
        """Test detection of command chaining."""
        queries_with_chaining = [
            "MATCH (n) RETURN n; CREATE (m)",
            "MATCH (n) RETURN n;",
            "; CREATE (n)",
            "MATCH (n); RETURN n",
        ]

        queries_without_chaining = [
            "MATCH (n) RETURN n",
        ]

        for query in queries_with_chaining:
            assert QueryValidator._has_command_chaining(
                query
            ), f"Query '{query}' should be detected as having command chaining"

        for query in queries_without_chaining:
            assert not QueryValidator._has_command_chaining(
                query
            ), f"Query '{query}' should not be detected as having command chaining"

        # Note: The current implementation is simple and detects ALL semicolons, even in strings
        # This is actually a reasonable security approach - being conservative


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_validate_cypher_query_function(self):
        """Test the validate_cypher_query convenience function."""
        # Valid query
        is_valid, error = validate_cypher_query("MATCH (n) RETURN n")
        assert is_valid
        assert error is None

        # Invalid query
        is_valid, error = validate_cypher_query("CREATE (n)")
        assert not is_valid
        assert "CREATE" in error

        # Empty query
        is_valid, error = validate_cypher_query("")
        assert not is_valid
        assert "Query cannot be empty" in error


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_case_insensitive_operation_detection(self):
        """Test that operation detection is case insensitive."""
        queries = [
            "create (n)",
            "CREATE (n)",
            "Create (n)",
            "cReAtE (n)",
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert not is_valid, f"Query '{query}' should be invalid"
            assert "CREATE" in error

    def test_multiline_queries(self):
        """Test validation of multiline queries."""
        valid_multiline = """
        MATCH (n:Person)
        WHERE n.age > 25
        RETURN n.name
        ORDER BY n.name
        LIMIT 10
        """

        is_valid, error = QueryValidator.validate_query(valid_multiline)
        assert is_valid, f"Multiline query should be valid, but got error: {error}"

        invalid_multiline = """
        MATCH (n:Person)
        WHERE n.age > 25
        CREATE (m:Evil)
        RETURN n.name
        """

        is_valid, error = QueryValidator.validate_query(invalid_multiline)
        assert not is_valid
        assert "CREATE" in error

    def test_whitespace_handling(self):
        """Test handling of various whitespace scenarios."""
        queries = [
            "MATCH   (n)   RETURN   n",  # Extra spaces
            "MATCH\t(n)\tRETURN\tn",  # Tabs
            "MATCH\n(n)\nRETURN\nn",  # Newlines
            "   MATCH (n) RETURN n   ",  # Leading/trailing whitespace
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert is_valid, f"Query '{query}' should be valid despite whitespace"

    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        queries = [
            "MATCH (n {name: 'José'}) RETURN n",
            "MATCH (n {name: '北京'}) RETURN n",
            "MATCH (n {name: 'Москва'}) RETURN n",
            "MATCH (n {emoji: '🚀'}) RETURN n",
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert is_valid, f"Query '{query}' with Unicode should be valid"


class TestPerformanceAndComplexity:
    """Test performance and complexity-related scenarios."""

    def test_deeply_nested_parentheses(self):
        """Test handling of deeply nested parentheses."""
        # Create deeply nested but balanced parentheses
        nested = "(" * 100 + "n" + ")" * 100
        query = f"MATCH {nested} RETURN n"

        is_valid, error = QueryValidator.validate_query(query)
        assert is_valid, "Deeply nested but balanced parentheses should be valid"

        # Test unbalanced deeply nested
        unbalanced = "(" * 100 + "n" + ")" * 99
        query = f"MATCH {unbalanced} RETURN n"

        is_valid, error = QueryValidator.validate_query(query)
        assert not is_valid
        assert "Unbalanced parentheses detected" in error

    def test_many_string_literals(self):
        """Test queries with many string literals."""
        # Create query with many string literals
        conditions = []
        for i in range(50):
            conditions.append(f"n.field{i} = 'value{i}'")

        query = f"MATCH (n) WHERE {' OR '.join(conditions)} RETURN n"

        is_valid, error = QueryValidator.validate_query(query)
        assert is_valid, "Query with many string literals should be valid"

    def test_complex_regex_patterns(self):
        """Test that regex patterns work correctly with complex cases."""
        # Test that operation detection works with surrounding text
        # Note: The current validator is conservative and will catch CREATE even in comments
        # This is actually good for security
        queries = [
            # Should be valid - CREATE in string literal
            "MATCH (n {description: 'This text has CREATE in it'}) RETURN n",
        ]

        for query in queries:
            is_valid, error = QueryValidator.validate_query(query)
            # These should be valid because CREATE is in strings (after string removal)
            assert is_valid, f"Query '{query}' should be valid, but got error: {error}"

        # Test queries that should be invalid due to comments containing operations
        invalid_queries = [
            "// This comment mentions CREATE but it's not a real operation\nMATCH (n) RETURN n",
            "MATCH (n) WHERE n.text =~ .*CREATE.* RETURN n",  # CREATE outside of quotes
        ]

        for query in invalid_queries:
            is_valid, error = QueryValidator.validate_query(query)
            # These should be invalid - the validator is conservative about CREATE
            assert not is_valid, f"Query '{query}' should be invalid (conservative approach)"

        # Test valid queries where CREATE is properly inside strings
        valid_queries = [
            # CREATE inside quotes - should be valid
            "MATCH (n) WHERE n.text =~ '.*CREATE.*' RETURN n",
        ]

        for query in valid_queries:
            is_valid, error = QueryValidator.validate_query(query)
            assert is_valid, f"Query '{query}' should be valid, but got error: {error}"


if __name__ == "__main__":
    pytest.main([__file__])
