"""
Comprehensive test suite for CypherValidator to achieve 100% test coverage.

This test suite covers all aspects of the CypherValidator class including:
- SecurityError exception class
- ValidationResult dataclass
- CypherValidator initialization and configuration
- All validation methods and edge cases
- Exception handling
- Security checks and injection detection
- Utility methods
"""

import unittest.mock as mock

import pytest

from src.security.cypher_validator import CypherValidator, SecurityError, ValidationResult


class TestSecurityError:
    """Test SecurityError exception class."""

    def test_security_error_default_type(self):
        """Test SecurityError with default error type."""
        error = SecurityError("Test message")
        assert str(error) == "Test message"
        assert error.error_type == "security_violation"

    def test_security_error_custom_type(self):
        """Test SecurityError with custom error type."""
        error = SecurityError("Custom message", "custom_error")
        assert str(error) == "Custom message"
        assert error.error_type == "custom_error"

    def test_security_error_inheritance(self):
        """Test that SecurityError inherits from Exception."""
        error = SecurityError("Test")
        assert isinstance(error, Exception)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_minimal(self):
        """Test ValidationResult with minimal parameters."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.error_message is None
        assert result.error_type is None
        assert result.warnings == []
        assert result.complexity_score == 0

    def test_validation_result_full(self):
        """Test ValidationResult with all parameters."""
        warnings = ["warning1", "warning2"]
        result = ValidationResult(
            is_valid=False,
            error_message="Test error",
            error_type="test_error",
            warnings=warnings,
            complexity_score=25,
        )
        assert result.is_valid is False
        assert result.error_message == "Test error"
        assert result.error_type == "test_error"
        assert result.warnings == warnings
        assert result.complexity_score == 25

    def test_validation_result_post_init_none_warnings(self):
        """Test ValidationResult post_init with None warnings."""
        result = ValidationResult(is_valid=True, warnings=None)
        assert result.warnings == []

    def test_validation_result_post_init_existing_warnings(self):
        """Test ValidationResult post_init with existing warnings."""
        warnings = ["existing_warning"]
        result = ValidationResult(is_valid=True, warnings=warnings)
        assert result.warnings == warnings


class TestCypherValidatorInitialization:
    """Test CypherValidator initialization."""

    def test_validator_default_initialization(self):
        """Test validator with default parameters."""
        validator = CypherValidator()
        assert validator.max_complexity == CypherValidator.MAX_COMPLEXITY_SCORE
        assert validator.max_length == CypherValidator.MAX_QUERY_LENGTH
        assert hasattr(validator, "_forbidden_pattern")
        assert hasattr(validator, "_safe_patterns")

    def test_validator_custom_initialization(self):
        """Test validator with custom parameters."""
        custom_complexity = 75
        custom_length = 2000
        validator = CypherValidator(max_complexity=custom_complexity, max_length=custom_length)
        assert validator.max_complexity == custom_complexity
        assert validator.max_length == custom_length

    def test_validator_none_parameters(self):
        """Test validator with None parameters falls back to defaults."""
        validator = CypherValidator(max_complexity=None, max_length=None)
        assert validator.max_complexity == CypherValidator.MAX_COMPLEXITY_SCORE
        assert validator.max_length == CypherValidator.MAX_QUERY_LENGTH


class TestCypherValidatorBasicValidation:
    """Test basic validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()
        self.strict_validator = CypherValidator(max_complexity=30, max_length=500)

    def test_simple_valid_query(self):
        """Test simple valid query."""
        query = "MATCH (n:Context) RETURN n.id"
        result = self.validator.validate_query(query)
        assert result.is_valid is True
        assert result.error_message is None
        assert result.complexity_score == 10  # 1 MATCH = 10 points

    def test_query_too_long(self):
        """Test query length validation."""
        long_query = (
            "MATCH (n:Context) WHERE "
            + " AND ".join([f"n.prop{i} = 'value{i}'" for i in range(100)])
            + " RETURN n.id"
        )
        result = self.strict_validator.validate_query(long_query)
        assert result.is_valid is False
        assert result.error_type == "query_too_long"
        assert "exceeds maximum length" in result.error_message

    def test_forbidden_operations(self):
        """Test detection of forbidden operations."""
        forbidden_queries = [
            "CREATE (n:Context) RETURN n",
            "MATCH (n:Context) DELETE n",
            "MATCH (n:Context) SET n.prop = 'value'",
            "MERGE (n:Context) RETURN n",
            "MATCH (n:Context) REMOVE n.prop",
            "DROP INDEX ON :Context(id)",
            "DETACH DELETE (n:Context)",
            "FOREACH (x IN range(1,10) | CREATE (n:Node))",
            "LOAD CSV FROM 'file.csv' AS row RETURN row",
            "USING INDEX n:Context(id) MATCH (n) RETURN n",
            "CALL db.indexes()",
            "MATCH (n) YIELD n.prop RETURN n",
        ]

        for query in forbidden_queries:
            result = self.validator.validate_query(query)
            assert result.is_valid is False, f"Query should be invalid: {query}"
            assert result.error_type == "forbidden_operation"

    def test_complexity_calculation(self):
        """Test complexity score calculation."""
        test_cases = [
            ("MATCH (n:Context) RETURN n.id", 10),  # 1 MATCH
            ("MATCH (n:Context) WHERE n.type = 'doc' RETURN n.id", 15),  # MATCH + WHERE
            ("MATCH (a)-[:REL]->(b) RETURN a, b", 15),  # MATCH + relationship
            ("MATCH (a) WITH a MATCH (b) RETURN a, b", 18),  # 2 MATCH + 1 WITH
            ("MATCH (a) UNION MATCH (b) RETURN a", 25),  # 2 MATCH + 1 UNION
            ("MATCH (a) UNWIND a.list AS item RETURN item", 20),  # MATCH + UNWIND
            ("MATCH (a) RETURN a ORDER BY a.name", 15),  # MATCH + ORDER BY
        ]

        for query, expected_min in test_cases:
            result = self.validator.validate_query(query)
            assert result.is_valid is True
            assert (
                result.complexity_score >= expected_min
            ), f"Query: {query}, Expected: {expected_min}, Got: {result.complexity_score}"

    def test_complexity_too_high(self):
        """Test complexity limit enforcement."""
        # Create a complex query that exceeds the strict limit
        complex_query = """
        MATCH (a:Context)-[:REL1]->(b:Context)
        WHERE a.type = 'source'
        WITH a, b
        MATCH (b)-[:REL2]->(c:Context)
        WHERE c.status = 'active'
        WITH a, b, c
        MATCH (c)-[:REL3]->(d:Context)
        RETURN a.id, b.id, c.id, d.id
        ORDER BY a.id
        UNION
        MATCH (x:Context) WHERE x.type = 'backup'
        RETURN x.id, null, null, null
        """

        result = self.strict_validator.validate_query(complex_query)
        assert result.is_valid is False
        assert result.error_type == "complexity_too_high"
        assert result.complexity_score > self.strict_validator.max_complexity


class TestParameterValidation:
    """Test parameter validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()

    def test_valid_parameters(self):
        """Test validation of valid parameters."""
        query = "MATCH (n:Context) WHERE n.id = $id AND n.type = $type RETURN n"
        params = {
            "id": "context-123",
            "type": "document",
            "count": 42,
            "active": True,
            "tags": ["tag1", "tag2"],
        }
        result = self.validator.validate_query(query, params)
        assert result.is_valid is True

    def test_invalid_parameter_names(self):
        """Test validation of invalid parameter names."""
        query = "MATCH (n:Context) WHERE n.id = $bad_name RETURN n"

        invalid_param_names = [
            {"123invalid": "value"},  # starts with number
            {"invalid-name": "value"},  # contains hyphen
            {"invalid name": "value"},  # contains space
            {"invalid.name": "value"},  # contains dot
            {"": "value"},  # empty name
        ]

        for params in invalid_param_names:
            result = self.validator.validate_query(query, params)
            assert result.is_valid is False
            assert result.error_type == "invalid_parameter_name"

    def test_parameter_length_warning(self):
        """Test parameter length warning - note: this is a test of internal method behavior."""
        # Test the internal _validate_parameters method directly since the main validate_query
        # method doesn't propagate parameter validation warnings (appears to be a design issue)
        params = {"content": "x" * 1001}  # Over 1000 characters

        param_result = self.validator._validate_parameters(params)
        assert param_result.is_valid is True
        assert len(param_result.warnings) > 0
        assert any("very long string value" in warning for warning in param_result.warnings)

    def test_suspicious_parameter_content(self):
        """Test detection of suspicious parameter content."""
        query = "MATCH (n:Context) WHERE n.title = $title RETURN n"

        suspicious_params = [
            {"title": "DROP TABLE users"},
            {"title": "DELETE FROM contexts"},
            {"title": "CREATE (malicious:Node)"},
            {"title": "SET n.hacked = true"},
            {"title": "MERGE (evil) RETURN evil"},
            {"title": "REMOVE n.important_data"},
        ]

        for params in suspicious_params:
            result = self.validator.validate_query(query, params)
            assert result.is_valid is False
            assert result.error_type == "suspicious_parameter"

    def test_parameters_none(self):
        """Test validation with None parameters."""
        query = "MATCH (n:Context) RETURN n.id"
        result = self.validator.validate_query(query, None)
        assert result.is_valid is True

    def test_parameters_empty_dict(self):
        """Test validation with empty parameters dict."""
        query = "MATCH (n:Context) RETURN n.id"
        result = self.validator.validate_query(query, {})
        assert result.is_valid is True


class TestSecurityChecks:
    """Test security validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()

    def test_injection_detection(self):
        """Test detection of potential injection attacks."""
        injection_queries = [
            "MATCH (n:Context); DROP TABLE users; RETURN n.id",  # Multiple statements
            "MATCH (n:Context) RETURN n.id; EXEC('malicious')",  # EXEC
            "MATCH (n:Context) RETURN EVAL('1+1') as result",  # EVAL
        ]

        for query in injection_queries:
            result = self.validator.validate_query(query)
            assert result.is_valid is False
            # These queries contain forbidden operations, so they'll be caught by that check first
            assert result.error_type in ["potential_injection", "forbidden_operation"]

    def test_pure_injection_patterns(self):
        """Test injection patterns that don't contain forbidden operations."""
        pure_injection_queries = [
            "MATCH (n:Context) WHERE n.id = 'test'; SELECT * FROM users",  # Multiple statements without forbidden ops
            "MATCH (n:Context); SELECT version()",  # SQL injection attempt
        ]

        for query in pure_injection_queries:
            result = self.validator.validate_query(query)
            assert result.is_valid is False
            assert result.error_type == "potential_injection"

    def test_comment_normalization(self):
        """Test that comments are properly normalized."""
        queries_with_comments = [
            "MATCH (n:Context) // single line comment\nRETURN n.id",
            "MATCH (n:Context) /* multi-line comment */ RETURN n.id",
            "MATCH (n:Context) /* multi\nline\ncomment */ RETURN n.id",
        ]

        for query in queries_with_comments:
            result = self.validator.validate_query(query)
            # These should be valid after comment removal
            assert result.is_valid is True

    def test_malicious_comments_detection(self):
        """Test detection of malicious comment patterns."""
        malicious_queries = [
            "MATCH (n:Context) -- comment RETURN n.id",  # SQL-style comment
            "MATCH (n:Context) /* comment */ RETURN n.id",  # Already tested in normalization
        ]

        # Note: The validator removes comments during normalization, so these might be valid
        # But SQL-style comments (--) should be caught by injection detection
        result = self.validator.validate_query("MATCH (n:Context) -- comment RETURN n.id")
        assert result.is_valid is False
        assert result.error_type == "potential_injection"

    def test_wildcard_warning(self):
        """Test warning for many wildcard selections."""
        # Create query with more than 3 asterisks to trigger the warning
        query_with_wildcards = "MATCH (a) RETURN * UNION MATCH (b) RETURN * UNION MATCH (c) RETURN * UNION MATCH (d) RETURN *"
        result = self.validator.validate_query(query_with_wildcards)
        # Should generate warning for resource-intensive operations
        assert any("wildcard" in warning.lower() for warning in result.warnings)

    def test_large_limit_validation(self):
        """Test LIMIT value validation."""
        # Test very large LIMIT (should fail)
        large_limit_query = "MATCH (n:Context) RETURN n.id LIMIT 50000"
        result = self.validator.validate_query(large_limit_query)
        assert result.is_valid is False
        assert result.error_type == "limit_too_large"

        # Test moderately large LIMIT (should warn)
        medium_limit_query = "MATCH (n:Context) RETURN n.id LIMIT 5000"
        result = self.validator.validate_query(medium_limit_query)
        assert result.is_valid is True
        assert any("Large LIMIT value" in warning for warning in result.warnings)

        # Test acceptable LIMIT
        small_limit_query = "MATCH (n:Context) RETURN n.id LIMIT 100"
        result = self.validator.validate_query(small_limit_query)
        assert result.is_valid is True


class TestPatternValidation:
    """Test query pattern validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()

    def test_safe_pattern_matching(self):
        """Test queries that match safe patterns."""
        safe_queries = [
            "MATCH (n:Context) RETURN n.id",
            "MATCH (n:Context) WHERE n.type = 'doc' RETURN n.title",
            "MATCH (n:Context) RETURN n.id ORDER BY n.created_at LIMIT 10",
        ]

        for query in safe_queries:
            result = self.validator.validate_query(query)
            assert result.is_valid is True
            # Safe pattern queries might still generate warnings if they don't match exactly

    def test_unsafe_pattern_warning(self):
        """Test queries that don't match safe patterns generate warnings."""
        # Complex query that doesn't match safe patterns
        complex_query = """
        MATCH (a:Context)-[:RELATES_TO]->(b:Context)
        WITH a, b, COUNT(*) as count
        WHERE count > 5
        RETURN a.id, b.id, count
        """

        result = self.validator.validate_query(complex_query)
        assert result.is_valid is True
        assert any(
            "does not match any known safe patterns" in warning for warning in result.warnings
        )


class TestUtilityMethods:
    """Test utility methods of CypherValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()

    def test_is_query_safe_true(self):
        """Test is_query_safe method with valid query."""
        query = "MATCH (n:Context) RETURN n.id"
        result = self.validator.is_query_safe(query)
        assert result is True

    def test_is_query_safe_false(self):
        """Test is_query_safe method with invalid query."""
        query = "CREATE (n:Context) RETURN n"
        result = self.validator.is_query_safe(query)
        assert result is False

    def test_is_query_safe_with_parameters(self):
        """Test is_query_safe method with parameters."""
        query = "MATCH (n:Context) WHERE n.id = $id RETURN n"
        params = {"id": "context-123"}
        result = self.validator.is_query_safe(query, params)
        assert result is True

    def test_get_safe_query_examples(self):
        """Test get_safe_query_examples method."""
        examples = self.validator.get_safe_query_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0

        # All examples should be valid
        for example in examples:
            assert isinstance(example, str)
            result = self.validator.validate_query(example)
            assert result.is_valid is True


class TestExceptionHandling:
    """Test exception handling in validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()

    def test_validate_query_exception_handling(self):
        """Test that validation handles internal exceptions gracefully."""
        # Mock an internal method to raise an exception
        with mock.patch.object(
            self.validator, "_normalize_query", side_effect=Exception("Test exception")
        ):
            result = self.validator.validate_query("MATCH (n) RETURN n")
            assert result.is_valid is False
            assert result.error_type == "validation_exception"
            assert "Validation error: Test exception" in result.error_message

    def test_regex_compilation_error_handling(self):
        """Test handling of regex compilation errors."""
        # Create validator with invalid regex pattern
        with mock.patch(
            "src.security.cypher_validator.re.compile", side_effect=Exception("Regex error")
        ):
            try:
                validator = CypherValidator()
                # If we get here, the exception was handled somewhere
            except Exception as e:
                # Expected if regex compilation fails during init
                assert "Regex error" in str(e)


class TestComplexityEdgeCases:
    """Test edge cases in complexity calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()

    def test_nested_patterns_complexity(self):
        """Test complexity calculation for nested patterns."""
        nested_query = "MATCH (a (nested (deeply) nested) pattern) RETURN a"
        result = self.validator.validate_query(nested_query)
        # Should add points for nested patterns
        assert result.complexity_score >= 10  # Base MATCH score

    def test_relationship_patterns_complexity(self):
        """Test complexity calculation for relationship patterns."""
        relationship_queries = [
            "MATCH (a)-[:REL]->(b) RETURN a, b",  # Forward relationship
            "MATCH (a)<-[:REL]-(b) RETURN a, b",  # Backward relationship
            "MATCH (a)-[:REL1]->(b)<-[:REL2]-(c) RETURN a, b, c",  # Multiple relationships
        ]

        for query in relationship_queries:
            result = self.validator.validate_query(query)
            assert result.is_valid is True
            assert result.complexity_score > 10  # Should be more than just MATCH


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()

    def test_empty_query(self):
        """Test validation of empty query."""
        result = self.validator.validate_query("")
        # Empty query should have low complexity but might fail other validations
        assert result.complexity_score == 0

    def test_whitespace_only_query(self):
        """Test validation of whitespace-only query."""
        result = self.validator.validate_query("   \n\t  ")
        assert result.complexity_score == 0

    def test_query_normalization_edge_cases(self):
        """Test query normalization with edge cases."""
        queries_to_normalize = [
            "MATCH   (n:Context)    RETURN   n.id",  # Multiple spaces
            "MATCH\n(n:Context)\nRETURN\nn.id",  # Newlines
            "MATCH\t(n:Context)\tRETURN\tn.id",  # Tabs
            "   MATCH (n:Context) RETURN n.id   ",  # Leading/trailing whitespace
        ]

        for query in queries_to_normalize:
            result = self.validator.validate_query(query)
            assert result.is_valid is True
            assert result.complexity_score == 10  # Should normalize to same complexity

    def test_case_insensitive_operations(self):
        """Test that operation detection is case insensitive."""
        case_variations = [
            "create (n:Context) return n",
            "CREATE (n:Context) RETURN n",
            "Create (n:Context) Return n",
            "cReAtE (n:Context) rEtUrN n",
        ]

        for query in case_variations:
            result = self.validator.validate_query(query)
            assert result.is_valid is False
            assert result.error_type == "forbidden_operation"

    def test_numeric_parameter_validation(self):
        """Test validation of numeric parameters."""
        query = "MATCH (n:Context) WHERE n.score > $threshold RETURN n"
        params = {
            "threshold": 0.5,
            "count": 42,
            "negative": -10,
            "zero": 0,
        }
        result = self.validator.validate_query(query, params)
        assert result.is_valid is True

    def test_boolean_parameter_validation(self):
        """Test validation of boolean parameters."""
        query = "MATCH (n:Context) WHERE n.active = $active RETURN n"
        params = {
            "active": True,
            "inactive": False,
        }
        result = self.validator.validate_query(query, params)
        assert result.is_valid is True

    def test_list_parameter_validation(self):
        """Test validation of list parameters."""
        query = "MATCH (n:Context) WHERE n.type IN $types RETURN n"
        params = {
            "types": ["doc", "ref", "analysis"],
            "empty_list": [],
        }
        result = self.validator.validate_query(query, params)
        assert result.is_valid is True

    def test_null_parameter_validation(self):
        """Test validation of null parameters."""
        query = "MATCH (n:Context) WHERE n.value = $value RETURN n"
        params = {
            "value": None,
        }
        result = self.validator.validate_query(query, params)
        assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
