#!/usr/bin/env python3
"""
Comprehensive tests for cypher_validator.py to achieve high coverage.

This test suite covers:
- CypherValidator initialization and configuration
- Query validation with all security checks
- Forbidden operation detection
- Query complexity scoring
- Parameter validation and injection prevention
- Safe pattern validation
- Security checks and injection detection
- Edge cases and error handling
"""

from unittest.mock import patch

import pytest

from src.security.cypher_validator import CypherValidator, SecurityError, ValidationResult


class TestSecurityError:
    """Test SecurityError exception class."""

    def test_security_error_default(self):
        """Test SecurityError with default error type."""
        error = SecurityError("Test security error")

        assert str(error) == "Test security error"
        assert error.error_type == "security_violation"
        assert isinstance(error, Exception)

    def test_security_error_custom_type(self):
        """Test SecurityError with custom error type."""
        error = SecurityError("Custom error", "injection_attempt")

        assert str(error) == "Custom error"
        assert error.error_type == "injection_attempt"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_basic(self):
        """Test basic ValidationResult creation."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.error_message is None
        assert result.error_type is None
        assert result.warnings == []
        assert result.complexity_score == 0

    def test_validation_result_with_error(self):
        """Test ValidationResult with error information."""
        result = ValidationResult(
            is_valid=False,
            error_message="Query contains forbidden operations",
            error_type="forbidden_operation",
            complexity_score=25,
        )

        assert result.is_valid is False
        assert result.error_message == "Query contains forbidden operations"
        assert result.error_type == "forbidden_operation"
        assert result.warnings == []
        assert result.complexity_score == 25

    def test_validation_result_with_warnings(self):
        """Test ValidationResult with warnings list."""
        warnings = ["Warning 1", "Warning 2"]
        result = ValidationResult(is_valid=True, warnings=warnings)

        assert result.is_valid is True
        assert result.warnings == warnings

    def test_validation_result_warnings_post_init(self):
        """Test ValidationResult warnings initialized in __post_init__."""
        result = ValidationResult(is_valid=True, warnings=None)

        assert result.warnings == []

    def test_validation_result_all_fields(self):
        """Test ValidationResult with all fields set."""
        result = ValidationResult(
            is_valid=True,
            error_message="No errors",
            error_type="none",
            warnings=["Minor warning"],
            complexity_score=15,
        )

        assert result.is_valid is True
        assert result.error_message == "No errors"
        assert result.error_type == "none"
        assert result.warnings == ["Minor warning"]
        assert result.complexity_score == 15


class TestCypherValidator:
    """Test CypherValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a standard validator instance."""
        return CypherValidator()

    @pytest.fixture
    def custom_validator(self):
        """Create a validator with custom limits."""
        return CypherValidator(max_complexity=50, max_length=1000)

    def test_validator_initialization_default(self, validator):
        """Test validator initialization with default values."""
        assert validator.max_complexity == 100
        assert validator.max_length == 5000
        assert validator._forbidden_pattern is not None
        assert len(validator._safe_patterns) == 2

    def test_validator_initialization_custom(self, custom_validator):
        """Test validator initialization with custom values."""
        assert custom_validator.max_complexity == 50
        assert custom_validator.max_length == 1000

    def test_class_constants(self):
        """Test class constants are properly defined."""
        assert "CREATE" in CypherValidator.FORBIDDEN_OPERATIONS
        assert "DELETE" in CypherValidator.FORBIDDEN_OPERATIONS
        assert "MATCH" in CypherValidator.ALLOWED_OPERATIONS
        assert "RETURN" in CypherValidator.ALLOWED_OPERATIONS
        assert CypherValidator.MAX_COMPLEXITY_SCORE == 100
        assert CypherValidator.MAX_QUERY_LENGTH == 5000

    def test_validate_query_simple_success(self, validator):
        """Test successful validation of simple query."""
        query = "MATCH (n:Context) RETURN n.id"

        result = validator.validate_query(query)

        assert result.is_valid is True
        assert result.error_message is None
        assert result.complexity_score == 10  # One MATCH = 10 points

    def test_validate_query_with_parameters_success(self, validator):
        """Test successful query validation with parameters."""
        query = "MATCH (n:Context) WHERE n.type = $type RETURN n"
        parameters = {"type": "decision"}

        result = validator.validate_query(query, parameters)

        assert result.is_valid is True
        assert result.complexity_score == 15  # MATCH(10) + WHERE(5)

    def test_validate_query_too_long(self, custom_validator):
        """Test query validation fails for queries that are too long."""
        query = "MATCH (n) RETURN n " * 200  # Make it very long

        result = custom_validator.validate_query(query)

        assert result.is_valid is False
        assert result.error_type == "query_too_long"
        assert "exceeds maximum length" in result.error_message

    def test_validate_query_forbidden_operations(self, validator):
        """Test query validation fails for forbidden operations."""
        forbidden_queries = [
            "CREATE (n:Test) RETURN n",
            "MATCH (n) DELETE n",
            "MATCH (n) SET n.prop = 'value'",
            "MERGE (n:Test {id: 1}) RETURN n",
            "MATCH (n) REMOVE n.prop",
            "DROP INDEX test_index",
            "CALL db.schema.visualization()",
        ]

        for query in forbidden_queries:
            result = validator.validate_query(query)

            assert result.is_valid is False
            assert result.error_type == "forbidden_operation"
            assert "forbidden operations" in result.error_message

    def test_validate_query_high_complexity(self, custom_validator):
        """Test query validation fails for high complexity queries."""
        # Create a complex query that exceeds the custom limit of 50
        query = """
        MATCH (a)-[:REL1]->(b)-[:REL2]->(c)
        WHERE a.prop = 'test'
        WITH a, b, c
        MATCH (d)-[:REL3]->(e)
        WHERE d.prop = 'test2'
        UNION
        MATCH (f)-[:REL4]->(g)
        RETURN f, g
        ORDER BY f.created
        LIMIT 100
        """

        result = custom_validator.validate_query(query)

        assert result.is_valid is False
        assert result.error_type == "complexity_too_high"
        assert "complexity score" in result.error_message
        assert result.complexity_score > 50

    def test_validate_query_parameter_validation_failure(self, validator):
        """Test query validation fails for invalid parameters."""
        query = "MATCH (n) WHERE n.prop = $param RETURN n"
        parameters = {"invalid-param": "value"}  # Invalid parameter name

        result = validator.validate_query(query, parameters)

        assert result.is_valid is False
        assert result.error_type == "invalid_parameter_name"
        assert "invalid-param" in result.error_message

    def test_validate_query_suspicious_parameter(self, validator):
        """Test query validation fails for suspicious parameter content."""
        query = "MATCH (n) WHERE n.prop = $param RETURN n"
        parameters = {"param": "DELETE FROM users"}  # Suspicious content

        result = validator.validate_query(query, parameters)

        assert result.is_valid is False
        assert result.error_type == "suspicious_parameter"
        assert "suspicious content" in result.error_message

    def test_validate_query_security_checks_failure(self, validator):
        """Test query validation fails for security threats."""
        malicious_queries = [
            (
                "MATCH (n) RETURN n; SELECT * FROM users",
                "potential_injection",
            ),  # Multiple statements
            ("MATCH (n) /* comment */ RETURN n", "potential_injection"),  # SQL-style comments
            ("MATCH (n) -- comment", "potential_injection"),  # SQL-style comments
            ("MATCH (n) RETURN n LIMIT 50000", "limit_too_large"),  # Excessive limit
        ]

        for query, expected_error_type in malicious_queries:
            result = validator.validate_query(query)

            assert result.is_valid is False
            assert result.error_type == expected_error_type

    def test_validate_query_with_warnings(self, validator):
        """Test query validation with warnings but still valid."""
        query = "MATCH (n) RETURN * LIMIT 5000"  # Large limit, many wildcards

        result = validator.validate_query(query)

        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("Large LIMIT value" in warning for warning in result.warnings)

    def test_validate_query_exception_handling(self, validator):
        """Test query validation handles exceptions gracefully."""
        with patch.object(validator, "_normalize_query", side_effect=Exception("Test error")):
            result = validator.validate_query("MATCH (n) RETURN n")

            assert result.is_valid is False
            assert result.error_type == "validation_exception"
            assert "Validation error" in result.error_message

    def test_normalize_query_comments(self, validator):
        """Test query normalization removes comments."""
        query_with_comments = """
        MATCH (n) // This is a comment
        /* Multi-line
           comment */
        RETURN n
        """

        normalized = validator._normalize_query(query_with_comments)

        assert "//" not in normalized
        assert "/*" not in normalized
        assert "*/" not in normalized
        assert "MATCH (n) RETURN n" == normalized

    def test_normalize_query_whitespace(self, validator):
        """Test query normalization handles whitespace."""
        query_with_whitespace = "   MATCH    (n)   \n\t  RETURN   n   "

        normalized = validator._normalize_query(query_with_whitespace)

        assert normalized == "MATCH (n) RETURN n"

    def test_check_forbidden_operations_clean(self, validator):
        """Test forbidden operations check with clean query."""
        clean_query = "MATCH (n:Context) WHERE n.type = 'test' RETURN n"

        result = validator._check_forbidden_operations(clean_query)

        assert result.is_valid is True

    def test_check_forbidden_operations_multiple(self, validator):
        """Test forbidden operations detection with multiple operations."""
        dirty_query = "CREATE (n) SET n.prop = 'value' DELETE n"

        result = validator._check_forbidden_operations(dirty_query)

        assert result.is_valid is False
        assert "CREATE" in result.error_message
        assert "SET" in result.error_message
        assert "DELETE" in result.error_message

    def test_calculate_complexity_basic(self, validator):
        """Test complexity calculation for basic operations."""
        query = "MATCH (n) WHERE n.prop = 'test' RETURN n"

        complexity = validator._calculate_complexity(query)

        assert complexity == 15  # MATCH(10) + WHERE(5)

    def test_calculate_complexity_advanced(self, validator):
        """Test complexity calculation for advanced operations."""
        query = """
        MATCH (a)-[:REL]->(b)
        WHERE a.prop = 'test'
        WITH a, b
        UNION
        MATCH (c)-[:REL2]->(d)
        UNWIND [1,2,3] AS x
        RETURN c, d
        ORDER BY c.created
        """

        complexity = validator._calculate_complexity(query)

        # MATCH(10) + WHERE(5) + WITH(8) + UNION(15) + MATCH(10) + UNWIND(10) + ORDER BY(5) + REL patterns(10)
        expected_min = 60
        assert complexity >= expected_min

    def test_calculate_complexity_nested_patterns(self, validator):
        """Test complexity calculation for nested patterns."""
        query = "MATCH ((a)-[:REL]->(b))-[:OUTER]->((c)-[:INNER]->(d)) RETURN a"

        complexity = validator._calculate_complexity(query)

        # Should include nested pattern scoring (20 points each)
        assert complexity > 10  # More than just basic MATCH

    def test_calculate_complexity_relationship_patterns(self, validator):
        """Test complexity calculation for relationship patterns."""
        query = "MATCH (a)-[:REL1]->(b)<-[:REL2]-(c) RETURN a"

        complexity = validator._calculate_complexity(query)

        # MATCH(10) + two relationship patterns(10)
        assert complexity == 20

    def test_validate_parameters_valid(self, validator):
        """Test parameter validation with valid parameters."""
        parameters = {"user_id": "123", "type": "context", "limit": 10, "active": True}

        result = validator._validate_parameters(parameters)

        assert result.is_valid is True
        assert result.warnings == []

    def test_validate_parameters_invalid_name(self, validator):
        """Test parameter validation with invalid parameter names."""
        invalid_params = [
            {"123invalid": "value"},  # Starts with number
            {"param-name": "value"},  # Contains hyphen
            {"param.name": "value"},  # Contains dot
            {"": "value"},  # Empty name
        ]

        for params in invalid_params:
            result = validator._validate_parameters(params)

            assert result.is_valid is False
            assert result.error_type == "invalid_parameter_name"

    def test_validate_parameters_long_string_warning(self, validator):
        """Test parameter validation generates warning for long strings."""
        parameters = {"param": "x" * 1500}  # Very long string

        result = validator._validate_parameters(parameters)

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert "very long string value" in result.warnings[0]

    def test_validate_parameters_suspicious_content(self, validator):
        """Test parameter validation detects suspicious content."""
        suspicious_values = [
            "DROP TABLE users",
            "CREATE INDEX test",
            "DELETE FROM table",
            "SET global.var = 1",
            "MERGE INTO table",
            "REMOVE property",
        ]

        for value in suspicious_values:
            result = validator._validate_parameters({"param": value})

            assert result.is_valid is False
            assert result.error_type == "suspicious_parameter"

    def test_validate_against_patterns_safe(self, validator):
        """Test validation against safe patterns with matching query."""
        safe_queries = [
            "MATCH (n:Context) RETURN n.id",
            "MATCH (n:Context) WHERE n.type = 'test' RETURN n.title",
            "MATCH (n:Context) RETURN n ORDER BY n.created LIMIT 10",
        ]

        for query in safe_queries:
            normalized = validator._normalize_query(query)
            result = validator._validate_against_patterns(normalized)

            assert result.is_valid is True
            # May or may not have warnings depending on exact pattern matching

    def test_validate_against_patterns_no_match(self, validator):
        """Test validation against safe patterns with non-matching query."""
        complex_query = "MATCH (a)-[:REL]->(b) WITH a, b MATCH (c) RETURN a, b, c"

        result = validator._validate_against_patterns(complex_query)

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert "does not match any known safe patterns" in result.warnings[0]

    def test_perform_security_checks_clean(self, validator):
        """Test security checks with clean query."""
        clean_query = "MATCH (n:Context) RETURN n.id LIMIT 10"

        result = validator._perform_security_checks(clean_query)

        assert result.is_valid is True

    def test_perform_security_checks_injection_patterns(self, validator):
        """Test security checks detect injection patterns."""
        injection_queries = [
            "MATCH (n); DROP DATABASE test",  # Multiple statements
            "MATCH (n) /* malicious */ RETURN n",  # Block comments
            "MATCH (n) -- comment",  # Line comments
            "EXEC some_procedure()",  # Dynamic execution
            "EVAL malicious_code",  # Dynamic execution
        ]

        for query in injection_queries:
            result = validator._perform_security_checks(query)

            assert result.is_valid is False
            assert result.error_type == "potential_injection"

    def test_perform_security_checks_wildcard_warning(self, validator):
        """Test security checks generate warning for many wildcards."""
        wildcard_query = "MATCH (*) RETURN *, *, *, *"  # Many wildcards

        result = validator._perform_security_checks(wildcard_query)

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert "many wildcard selections" in result.warnings[0]

    def test_perform_security_checks_limit_validation(self, validator):
        """Test security checks validate LIMIT values."""
        # Test large but acceptable limit
        large_limit_query = "MATCH (n) RETURN n LIMIT 5000"
        result = validator._perform_security_checks(large_limit_query)

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert "Large LIMIT value" in result.warnings[0]

        # Test excessive limit
        excessive_limit_query = "MATCH (n) RETURN n LIMIT 50000"
        result = validator._perform_security_checks(excessive_limit_query)

        assert result.is_valid is False
        assert result.error_type == "limit_too_large"

    def test_perform_security_checks_multiple_limits(self, validator):
        """Test security checks with multiple LIMIT clauses."""
        query = "MATCH (n) RETURN n LIMIT 100 UNION MATCH (m) RETURN m LIMIT 2000"

        result = validator._perform_security_checks(query)

        assert result.is_valid is True
        assert any("Large LIMIT value: 2000" in warning for warning in result.warnings)

    def test_is_query_safe_success(self, validator):
        """Test is_query_safe convenience method with safe query."""
        safe_query = "MATCH (n:Context) RETURN n.id LIMIT 10"

        assert validator.is_query_safe(safe_query) is True

    def test_is_query_safe_failure(self, validator):
        """Test is_query_safe convenience method with unsafe query."""
        unsafe_query = "CREATE (n:Test) RETURN n"

        assert validator.is_query_safe(unsafe_query) is False

    def test_is_query_safe_with_parameters(self, validator):
        """Test is_query_safe with parameters."""
        query = "MATCH (n) WHERE n.type = $type RETURN n"
        safe_params = {"type": "context"}
        unsafe_params = {"type": "DELETE FROM users"}

        assert validator.is_query_safe(query, safe_params) is True
        assert validator.is_query_safe(query, unsafe_params) is False

    def test_get_safe_query_examples(self, validator):
        """Test get_safe_query_examples method."""
        examples = validator.get_safe_query_examples()

        assert isinstance(examples, list)
        assert len(examples) > 0

        # All examples should be safe
        for example in examples:
            assert validator.is_query_safe(example) is True

    def test_get_safe_query_examples_content(self, validator):
        """Test get_safe_query_examples returns expected content."""
        examples = validator.get_safe_query_examples()

        assert any("MATCH (n:Context)" in example for example in examples)
        assert any("WHERE" in example for example in examples)
        assert any("LIMIT" in example for example in examples)
        assert any("COUNT" in example for example in examples)

    def test_validation_with_none_parameters(self, validator):
        """Test validation works correctly with None parameters."""
        query = "MATCH (n) RETURN n"

        result = validator.validate_query(query, None)

        assert result.is_valid is True

    def test_validation_with_empty_parameters(self, validator):
        """Test validation works correctly with empty parameters."""
        query = "MATCH (n) RETURN n"

        result = validator.validate_query(query, {})

        assert result.is_valid is True

    def test_case_insensitive_forbidden_operations(self, validator):
        """Test forbidden operations detection is case insensitive."""
        queries = [
            "create (n:Test) return n",
            "CREATE (n:Test) RETURN n",
            "Create (n:Test) Return n",
            "cReAtE (n:Test) rEtUrN n",
        ]

        for query in queries:
            result = validator.validate_query(query)

            assert result.is_valid is False
            assert result.error_type == "forbidden_operation"

    def test_case_insensitive_complexity_calculation(self, validator):
        """Test complexity calculation is case insensitive."""
        queries = [
            "MATCH (n) WHERE n.prop = 'test' RETURN n",
            "match (n) where n.prop = 'test' return n",
            "Match (n) Where n.prop = 'test' Return n",
        ]

        complexities = [validator._calculate_complexity(q) for q in queries]

        # All should have the same complexity score
        assert all(c == complexities[0] for c in complexities)
        assert complexities[0] == 15  # MATCH(10) + WHERE(5)

    @patch("security.cypher_validator.logger")
    def test_logging_success(self, mock_logger, validator):
        """Test that successful validation logs appropriately."""
        query = "MATCH (n) RETURN n"

        validator.validate_query(query)

        mock_logger.info.assert_called_once()
        assert "Query validation passed" in mock_logger.info.call_args[0][0]

    @patch("security.cypher_validator.logger")
    def test_logging_exception(self, mock_logger, validator):
        """Test that validation exceptions are logged."""
        with patch.object(validator, "_normalize_query", side_effect=Exception("Test error")):
            validator.validate_query("MATCH (n) RETURN n")

            mock_logger.error.assert_called_once()
            assert "Query validation failed" in mock_logger.error.call_args[0][0]

    def test_edge_case_empty_query(self, validator):
        """Test validation with empty query."""
        result = validator.validate_query("")

        assert result.is_valid is True
        assert result.complexity_score == 0

    def test_edge_case_whitespace_only_query(self, validator):
        """Test validation with whitespace-only query."""
        result = validator.validate_query("   \n\t   ")

        assert result.is_valid is True
        assert result.complexity_score == 0

    def test_edge_case_query_with_only_comments(self, validator):
        """Test validation with query containing only comments."""
        comment_query = "// This is just a comment\n/* And this too */"

        result = validator.validate_query(comment_query)

        # Comments are detected as potentially malicious patterns in security checks
        assert result.is_valid is False
        assert result.error_type == "potential_injection"


class TestSuspiciousCommentDetection:
    """Test suite for the new _contains_suspicious_comments method."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return CypherValidator()

    def test_legitimate_cypher_line_comments_allowed(self, validator):
        """Test that legitimate Cypher line comments are allowed."""
        queries = [
            "MATCH (n) // Find all nodes\nRETURN n",
            "MATCH (n) RETURN n // Simple query",
            "// This is a normal comment\nMATCH (n) RETURN n",
        ]

        for query in queries:
            result = validator.validate_query(query)
            assert result.is_valid is True, f"Query should be valid: {query}"

    def test_legitimate_short_block_comments_allowed(self, validator):
        """Test that legitimate short block comments are allowed."""
        queries = [
            "MATCH (n) /* get nodes */ RETURN n",
            "/* Query description */MATCH (n) RETURN n",
            "MATCH (n) RETURN n /* end */",
        ]

        for query in queries:
            result = validator.validate_query(query)
            assert result.is_valid is True, f"Query should be valid: {query}"

    def test_suspicious_block_comments_with_sql_keywords_blocked(self, validator):
        """Test that block comments containing SQL keywords are blocked."""
        suspicious_queries = [
            "MATCH (n) /* CREATE TABLE users */ RETURN n",
            "MATCH (n) /* DELETE FROM users */ RETURN n", 
            "MATCH (n) /* DROP TABLE users */ RETURN n",
            "MATCH (n) /* SET password = 'hacked' */ RETURN n",
            "MATCH (n) /* EXEC sp_configure */ RETURN n",
            "MATCH (n) /* UNION SELECT * FROM */ RETURN n",
            "MATCH (n) /* INSERT INTO users */ RETURN n",
        ]

        for query in suspicious_queries:
            result = validator.validate_query(query)
            assert result.is_valid is False, f"Query should be blocked: {query}"
            assert result.error_type == "potential_injection"

    def test_suspicious_block_comments_with_cypher_keywords_blocked(self, validator):
        """Test that block comments containing dangerous Cypher keywords are blocked."""
        suspicious_queries = [
            "MATCH (n) /* CREATE (x:User) */ RETURN n",
            "MATCH (n) /* MERGE (x:Admin) */ RETURN n",
            "MATCH (n) /* REMOVE n.sensitive */ RETURN n",
            "MATCH (n) /* DETACH DELETE n */ RETURN n",
            "MATCH (n) /* CALL db.schema() */ RETURN n",
        ]

        for query in suspicious_queries:
            result = validator.validate_query(query)
            assert result.is_valid is False, f"Query should be blocked: {query}"
            assert result.error_type == "potential_injection"

    def test_suspicious_query_termination_attempts_blocked(self, validator):
        """Test that comments containing query termination attempts are blocked."""
        suspicious_queries = [
            "MATCH (n) /* ; DROP TABLE users */ RETURN n",
            "MATCH (n) /* test; CREATE (x) */ RETURN n", 
            "MATCH (n) /* ; SELECT * FROM admin */ RETURN n",
        ]

        for query in suspicious_queries:
            result = validator.validate_query(query)
            assert result.is_valid is False, f"Query should be blocked: {query}"
            assert result.error_type == "potential_injection"

    def test_very_long_comments_blocked(self, validator):
        """Test that very long comments are blocked as suspicious."""
        long_comment = "x" * 150
        suspicious_query = f"MATCH (n) /* {long_comment} */ RETURN n"

        result = validator.validate_query(suspicious_query)
        assert result.is_valid is False
        assert result.error_type == "potential_injection"

    def test_hex_encoded_content_blocked(self, validator):
        """Test that comments with hex-encoded content are blocked."""
        hex_encoded = "414243444546" * 5  # Long hex string
        suspicious_query = f"MATCH (n) /* {hex_encoded} */ RETURN n"

        result = validator.validate_query(suspicious_query)
        assert result.is_valid is False
        assert result.error_type == "potential_injection"

    def test_base64_encoded_content_blocked(self, validator):
        """Test that comments with base64-encoded content are blocked."""
        base64_encoded = "QWxhZGRpbjpvcGVuIHNlc2FtZQ==" * 3  # Long base64 string
        suspicious_query = f"MATCH (n) /* {base64_encoded} */ RETURN n"

        result = validator.validate_query(suspicious_query)
        assert result.is_valid is False
        assert result.error_type == "potential_injection"

    def test_very_long_line_comments_blocked(self, validator):
        """Test that very long line comments are blocked."""
        long_comment = "x" * 250
        suspicious_query = f"MATCH (n) RETURN n // {long_comment}"

        result = validator.validate_query(suspicious_query)
        assert result.is_valid is False
        assert result.error_type == "potential_injection"

    def test_line_comments_with_multiple_statements_blocked(self, validator):
        """Test that line comments with multiple statements are blocked."""
        suspicious_queries = [
            "MATCH (n) RETURN n // test; CREATE; DROP;",
            "MATCH (n) RETURN n // one; two; three; four",
        ]

        for query in suspicious_queries:
            result = validator.validate_query(query)
            assert result.is_valid is False, f"Query should be blocked: {query}"
            assert result.error_type == "potential_injection"

    def test_mixed_legitimate_and_suspicious_comments(self, validator):
        """Test queries with both legitimate and suspicious comments."""
        # Legitimate part should not override suspicious part
        suspicious_query = (
            "// This is fine\n"
            "MATCH (n) /* DROP TABLE users */ RETURN n"
        )

        result = validator.validate_query(suspicious_query)
        assert result.is_valid is False
        assert result.error_type == "potential_injection"

    def test_case_insensitive_keyword_detection(self, validator):
        """Test that keyword detection in comments is case insensitive."""
        suspicious_queries = [
            "MATCH (n) /* create table */ RETURN n",
            "MATCH (n) /* CREATE TABLE */ RETURN n",
            "MATCH (n) /* Create Table */ RETURN n",
            "MATCH (n) /* cReAtE tAbLe */ RETURN n",
        ]

        for query in suspicious_queries:
            result = validator.validate_query(query)
            assert result.is_valid is False, f"Query should be blocked: {query}"
            assert result.error_type == "potential_injection"

    def test_legitimate_technical_comments_allowed(self, validator):
        """Test that legitimate technical comments are allowed."""
        legitimate_queries = [
            "MATCH (n) /* TODO: optimize this query */ RETURN n",
            "MATCH (n) /* BUG: fix indexing issue */ RETURN n",
            "MATCH (n) /* NOTE: performance testing needed */ RETURN n", 
            "MATCH (n) /* FIXME: refactor later */ RETURN n",
            "MATCH (n) /* Index: user_email_idx */ RETURN n",
        ]

        for query in legitimate_queries:
            result = validator.validate_query(query)
            assert result.is_valid is True, f"Query should be valid: {query}"

    def test_nested_block_comments_handled_correctly(self, validator):
        """Test handling of nested block comment patterns."""
        # Nested comments aren't valid Cypher, but we should handle them safely
        query_with_nested = "MATCH (n) /* outer /* inner */ outer */ RETURN n"
        
        result = validator.validate_query(query_with_nested)
        # Should not crash and should be processed safely
        assert isinstance(result.is_valid, bool)

    def test_comment_boundary_edge_cases(self, validator):
        """Test edge cases around comment boundaries."""
        edge_cases = [
            "MATCH (n) /**/ RETURN n",  # Empty block comment
            "MATCH (n) /**//**/ RETURN n",  # Multiple empty comments
            "MATCH (n) //\nRETURN n",  # Empty line comment
            "MATCH (n) /* * */ RETURN n",  # Comment with asterisk
        ]

        for query in edge_cases:
            result = validator.validate_query(query)
            # Should not crash
            assert isinstance(result.is_valid, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
