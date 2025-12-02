#!/usr/bin/env python3
"""
Test suite for security/cypher_validator.py - Cypher query validation tests
"""
import pytest
import re
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Import the module under test
from src.security.cypher_validator import (
    SecurityError,
    ValidationResult,
    CypherValidator
)


class TestSecurityError:
    """Test suite for SecurityError exception"""

    def test_security_error_creation_default(self):
        """Test SecurityError creation with default error type"""
        error = SecurityError("Test security violation")
        
        assert str(error) == "Test security violation"
        assert error.error_type == "security_violation"

    def test_security_error_creation_custom_type(self):
        """Test SecurityError creation with custom error type"""
        error = SecurityError("Custom error", "custom_violation")
        
        assert str(error) == "Custom error"
        assert error.error_type == "custom_violation"

    def test_security_error_inheritance(self):
        """Test that SecurityError inherits from Exception"""
        error = SecurityError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, SecurityError)


class TestValidationResult:
    """Test suite for ValidationResult dataclass"""

    def test_validation_result_creation_minimal(self):
        """Test ValidationResult creation with minimal parameters"""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert result.error_message is None
        assert result.error_type is None
        assert result.warnings == []  # Post-init creates empty list
        assert result.complexity_score == 0

    def test_validation_result_creation_complete(self):
        """Test ValidationResult creation with all parameters"""
        warnings = ["Warning 1", "Warning 2"]
        result = ValidationResult(
            is_valid=False,
            error_message="Test error",
            error_type="test_error",
            warnings=warnings,
            complexity_score=50
        )
        
        assert result.is_valid is False
        assert result.error_message == "Test error"
        assert result.error_type == "test_error"
        assert result.warnings == warnings
        assert result.complexity_score == 50

    def test_validation_result_post_init_warnings(self):
        """Test ValidationResult post_init creates empty warnings list"""
        result = ValidationResult(is_valid=True, warnings=None)
        
        assert result.warnings == []  # Should be converted from None

    def test_validation_result_equality(self):
        """Test ValidationResult equality comparison"""
        result1 = ValidationResult(is_valid=True, error_message="test")
        result2 = ValidationResult(is_valid=True, error_message="test")
        result3 = ValidationResult(is_valid=False, error_message="test")
        
        assert result1 == result2
        assert result1 != result3


class TestCypherValidatorInit:
    """Test suite for CypherValidator initialization"""

    def test_cypher_validator_default_init(self):
        """Test CypherValidator initialization with defaults"""
        validator = CypherValidator()
        
        assert validator.max_complexity == CypherValidator.MAX_COMPLEXITY_SCORE
        assert validator.max_length == CypherValidator.MAX_QUERY_LENGTH
        assert hasattr(validator, '_forbidden_pattern')
        assert hasattr(validator, '_safe_patterns')
        assert len(validator._safe_patterns) > 0

    def test_cypher_validator_custom_limits(self):
        """Test CypherValidator with custom limits"""
        validator = CypherValidator(max_complexity=200, max_length=1000)
        
        assert validator.max_complexity == 200
        assert validator.max_length == 1000

    def test_cypher_validator_forbidden_operations_constants(self):
        """Test that forbidden operations constants are properly defined"""
        forbidden = CypherValidator.FORBIDDEN_OPERATIONS
        allowed = CypherValidator.ALLOWED_OPERATIONS
        
        # Check key forbidden operations
        assert "CREATE" in forbidden
        assert "DELETE" in forbidden
        assert "SET" in forbidden
        assert "MERGE" in forbidden
        assert "DROP" in forbidden
        
        # Check key allowed operations
        assert "MATCH" in allowed
        assert "RETURN" in allowed
        assert "WHERE" in allowed
        assert "LIMIT" in allowed

    def test_cypher_validator_pattern_compilation(self):
        """Test that regex patterns are compiled correctly"""
        validator = CypherValidator()
        
        # Test forbidden pattern matches forbidden operations
        assert validator._forbidden_pattern.search("CREATE (n)") is not None
        assert validator._forbidden_pattern.search("DELETE n") is not None
        assert validator._forbidden_pattern.search("MATCH (n) RETURN n") is None


class TestCypherValidatorBasicValidation:
    """Test suite for basic query validation"""

    def test_validate_query_valid_simple(self):
        """Test validation of simple valid query"""
        validator = CypherValidator()
        query = "MATCH (n:Context) RETURN n.id LIMIT 10"
        
        result = validator.validate_query(query)
        
        assert result.is_valid is True
        assert result.error_message is None
        assert result.complexity_score >= 0

    def test_validate_query_too_long(self):
        """Test validation rejects overly long queries"""
        validator = CypherValidator(max_length=100)
        query = "MATCH (n:Context) RETURN n.id, n.title, n.content, n.created_at, n.updated_at" * 10
        
        result = validator.validate_query(query)
        
        assert result.is_valid is False
        assert result.error_type == "query_too_long"
        assert "exceeds maximum length" in result.error_message

    def test_validate_query_forbidden_operation_create(self):
        """Test validation rejects CREATE operations"""
        validator = CypherValidator()
        query = "CREATE (n:Context {title: 'test'})"
        
        result = validator.validate_query(query)
        
        assert result.is_valid is False
        assert result.error_type == "forbidden_operation"
        assert "CREATE" in result.error_message

    def test_validate_query_forbidden_operation_delete(self):
        """Test validation rejects DELETE operations"""
        validator = CypherValidator()
        query = "MATCH (n:Context) DELETE n"
        
        result = validator.validate_query(query)
        
        assert result.is_valid is False
        assert result.error_type == "forbidden_operation"
        assert "DELETE" in result.error_message

    def test_validate_query_forbidden_operation_multiple(self):
        """Test validation detects multiple forbidden operations"""
        validator = CypherValidator()
        query = "CREATE (n) SET n.prop = 'value' MERGE (m)"
        
        result = validator.validate_query(query)
        
        assert result.is_valid is False
        assert result.error_type == "forbidden_operation"
        # Should detect multiple operations
        assert any(op in result.error_message for op in ["CREATE", "SET", "MERGE"])

    def test_validate_query_case_insensitive_forbidden(self):
        """Test validation is case insensitive for forbidden operations"""
        validator = CypherValidator()
        queries = [
            "create (n:Test)",
            "Create (n:Test)", 
            "CREATE (n:Test)",
            "CrEaTe (n:Test)"
        ]
        
        for query in queries:
            result = validator.validate_query(query)
            assert result.is_valid is False
            assert result.error_type == "forbidden_operation"

    def test_validate_query_high_complexity(self):
        """Test validation rejects high complexity queries"""
        validator = CypherValidator(max_complexity=50)
        
        # Build a complex query with many MATCH clauses (10 points each)
        query = " ".join([
            "MATCH (n1) MATCH (n2) MATCH (n3) MATCH (n4) MATCH (n5)",
            "MATCH (n6) WHERE n1.id = 1 WHERE n2.id = 2", 
            "RETURN n1, n2, n3, n4, n5, n6"
        ])
        
        result = validator.validate_query(query)
        
        assert result.is_valid is False
        assert result.error_type == "complexity_too_high"
        assert "complexity score" in result.error_message
        assert result.complexity_score > 50

    def test_validate_query_with_parameters(self):
        """Test validation with valid parameters"""
        validator = CypherValidator()
        query = "MATCH (n:Context) WHERE n.type = $type RETURN n"
        parameters = {"type": "document"}
        
        result = validator.validate_query(query, parameters)
        
        assert result.is_valid is True

    def test_validate_query_empty_parameters(self):
        """Test validation with empty parameters"""
        validator = CypherValidator()
        query = "MATCH (n:Context) RETURN n LIMIT 5"
        
        result = validator.validate_query(query, {})
        
        assert result.is_valid is True


class TestCypherValidatorNormalization:
    """Test suite for query normalization"""

    def test_normalize_query_basic(self):
        """Test basic query normalization"""
        validator = CypherValidator()
        query = "  MATCH   (n:Context)    RETURN   n  "
        
        normalized = validator._normalize_query(query)
        
        assert normalized == "MATCH (n:Context) RETURN n"

    def test_normalize_query_single_line_comments(self):
        """Test normalization removes single line comments"""
        validator = CypherValidator()
        query = "MATCH (n) // This is a comment\nRETURN n"
        
        normalized = validator._normalize_query(query)
        
        assert "// This is a comment" not in normalized
        assert "MATCH (n)" in normalized
        assert "RETURN n" in normalized

    def test_normalize_query_multi_line_comments(self):
        """Test normalization removes multi-line comments"""
        validator = CypherValidator()
        query = "MATCH (n) /* This is a\nmulti-line comment */ RETURN n"
        
        normalized = validator._normalize_query(query)
        
        assert "/* This is a" not in normalized
        assert "multi-line comment */" not in normalized
        assert "MATCH (n)" in normalized
        assert "RETURN n" in normalized

    def test_normalize_query_multiple_whitespace(self):
        """Test normalization collapses multiple whitespace"""
        validator = CypherValidator()
        query = "MATCH\t\t(n:Context)\n\n\nRETURN     n"
        
        normalized = validator._normalize_query(query)
        
        assert normalized == "MATCH (n:Context) RETURN n"

    def test_normalize_query_complex_example(self):
        """Test normalization with complex real-world example"""
        validator = CypherValidator()
        query = """
        // Get contexts by type
        MATCH (n:Context) 
        /* Check type parameter */
        WHERE n.type = $type
        /* Return results with limit */
        RETURN n.id, n.title   LIMIT   10
        """
        
        normalized = validator._normalize_query(query)
        
        assert "// Get contexts by type" not in normalized
        assert "/* Check type parameter */" not in normalized
        assert "MATCH (n:Context) WHERE n.type = $type RETURN n.id, n.title LIMIT 10" == normalized


class TestCypherValidatorComplexity:
    """Test suite for complexity calculation"""

    def test_calculate_complexity_simple(self):
        """Test complexity calculation for simple query"""
        validator = CypherValidator()
        query = "MATCH (n) RETURN n"
        
        complexity = validator._calculate_complexity(query)
        
        # Should be 10 (1 MATCH * 10 points)
        assert complexity == 10

    def test_calculate_complexity_with_where(self):
        """Test complexity calculation with WHERE clause"""
        validator = CypherValidator()
        query = "MATCH (n) WHERE n.id = 1 RETURN n"
        
        complexity = validator._calculate_complexity(query)
        
        # Should be 15 (1 MATCH * 10 + 1 WHERE * 5)
        assert complexity == 15

    def test_calculate_complexity_multiple_clauses(self):
        """Test complexity calculation with multiple clauses"""
        validator = CypherValidator()
        query = "MATCH (n) WHERE n.id = 1 WITH n WHERE n.active = true RETURN n ORDER BY n.name"
        
        complexity = validator._calculate_complexity(query)
        
        # 1 MATCH(10) + 2 WHERE(10) + 1 WITH(8) + 1 ORDER BY(5) = 33
        assert complexity == 33

    def test_calculate_complexity_relationships(self):
        """Test complexity calculation with relationship patterns"""
        validator = CypherValidator()
        query = "MATCH (n)-[:RELATES_TO]->(m)<-[:CONNECTS]-(p) RETURN n, m, p"
        
        complexity = validator._calculate_complexity(query)
        
        # 1 MATCH(10) + 2 relationships(10) = 20
        assert complexity == 20

    def test_calculate_complexity_union(self):
        """Test complexity calculation with UNION"""
        validator = CypherValidator()
        query = "MATCH (n:Type1) RETURN n UNION MATCH (m:Type2) RETURN m"
        
        complexity = validator._calculate_complexity(query)
        
        # 2 MATCH(20) + 1 UNION(15) = 35
        assert complexity == 35

    def test_calculate_complexity_nested_patterns(self):
        """Test complexity calculation with nested patterns"""
        validator = CypherValidator()
        query = "MATCH (n:(Label1|Label2)) RETURN n"
        
        complexity = validator._calculate_complexity(query)
        
        # This should detect nested patterns and add complexity
        assert complexity >= 10  # At least the MATCH cost


class TestCypherValidatorParameterValidation:
    """Test suite for parameter validation"""

    def test_validate_parameters_valid(self):
        """Test validation of valid parameters"""
        validator = CypherValidator()
        parameters = {
            "type": "document",
            "limit_val": 10,
            "active": True,
            "scores": [1.0, 2.5, 3.7]
        }
        
        result = validator._validate_parameters(parameters)
        
        assert result.is_valid is True

    def test_validate_parameters_invalid_key_format(self):
        """Test validation rejects invalid parameter key formats"""
        validator = CypherValidator()
        invalid_keys = [
            {"1invalid": "value"},  # Starts with number
            {"invalid-key": "value"},  # Contains dash
            {"invalid.key": "value"},  # Contains dot
            {"invalid key": "value"}  # Contains space
        ]
        
        for params in invalid_keys:
            result = validator._validate_parameters(params)
            assert result.is_valid is False
            assert result.error_type == "invalid_parameter_name"

    def test_validate_parameters_valid_key_formats(self):
        """Test validation accepts valid parameter key formats"""
        validator = CypherValidator()
        valid_keys = [
            {"valid_key": "value"},
            {"ValidKey": "value"},
            {"_private": "value"},
            {"key123": "value"},
            {"a": "value"}
        ]
        
        for params in valid_keys:
            result = validator._validate_parameters(params)
            assert result.is_valid is True

    def test_validate_parameters_long_string_warning(self):
        """Test validation warns about very long string values"""
        validator = CypherValidator()
        parameters = {"long_param": "x" * 1500}
        
        result = validator._validate_parameters(parameters)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "very long string value" in result.warnings[0]

    def test_validate_parameters_suspicious_content(self):
        """Test validation detects suspicious parameter content"""
        validator = CypherValidator()
        suspicious_values = [
            "DROP TABLE users",
            "DELETE FROM contexts",
            "CREATE INDEX malicious",
            "SET password = 'hacked'",
            "MERGE (evil)",
            "REMOVE n.security"
        ]
        
        for value in suspicious_values:
            parameters = {"param": value}
            result = validator._validate_parameters(parameters)
            
            assert result.is_valid is False
            assert result.error_type == "suspicious_parameter"
            assert "suspicious content" in result.error_message

    def test_validate_parameters_case_insensitive_suspicious(self):
        """Test validation detects suspicious content case-insensitively"""
        validator = CypherValidator()
        suspicious_values = [
            "drop table users",
            "Delete From contexts", 
            "CREATE index malicious",
            "set password = 'hacked'"
        ]
        
        for value in suspicious_values:
            parameters = {"param": value}
            result = validator._validate_parameters(parameters)
            
            assert result.is_valid is False
            assert result.error_type == "suspicious_parameter"


class TestCypherValidatorSecurityChecks:
    """Test suite for additional security checks"""

    def test_security_checks_valid_query(self):
        """Test security checks pass for valid query"""
        validator = CypherValidator()
        query = "MATCH (n:Context) RETURN n LIMIT 100"
        
        result = validator._perform_security_checks(query)
        
        assert result.is_valid is True

    def test_security_checks_multiple_statements(self):
        """Test security checks detect multiple statements"""
        validator = CypherValidator()
        query = "MATCH (n) RETURN n; DROP DATABASE"
        
        result = validator._perform_security_checks(query)
        
        assert result.is_valid is False
        assert result.error_type == "potential_injection"

    def test_security_checks_sql_comments(self):
        """Test security checks detect SQL-style comments"""
        validator = CypherValidator()
        query = "MATCH (n) -- malicious comment"
        
        result = validator._perform_security_checks(query)
        
        assert result.is_valid is False
        assert result.error_type == "potential_injection"

    def test_security_checks_exec_patterns(self):
        """Test security checks detect EXEC/EVAL patterns"""
        validator = CypherValidator()
        malicious_queries = [
            "MATCH (n) EXEC('malicious')",
            "EVAL some_function()",
            "match (n) exec malicious"
        ]
        
        for query in malicious_queries:
            result = validator._perform_security_checks(query)
            assert result.is_valid is False
            assert result.error_type == "potential_injection"

    def test_security_checks_large_limit_error(self):
        """Test security checks reject very large LIMIT values"""
        validator = CypherValidator()
        query = "MATCH (n) RETURN n LIMIT 50000"
        
        result = validator._perform_security_checks(query)
        
        assert result.is_valid is False
        assert result.error_type == "limit_too_large"
        assert "50000" in result.error_message

    def test_security_checks_large_limit_warning(self):
        """Test security checks warn about moderately large LIMIT values"""
        validator = CypherValidator()
        query = "MATCH (n) RETURN n LIMIT 5000"
        
        result = validator._perform_security_checks(query)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "Large LIMIT value" in result.warnings[0]

    def test_security_checks_many_wildcards_warning(self):
        """Test security checks warn about many wildcard selections"""
        validator = CypherValidator()
        query = "MATCH (n) RETURN n.*, m.*, p.*, q.*"
        
        result = validator._perform_security_checks(query)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "wildcard selections" in result.warnings[0]

    def test_security_checks_suspicious_comment_patterns(self):
        """Test security checks detect suspicious comment patterns"""
        validator = CypherValidator()
        suspicious_queries = [
            "MATCH (n) /* comment ; malicious */ RETURN n",
            "/* block1 */ /* block2 */ MATCH (n)"
        ]
        
        for query in suspicious_queries:
            result = validator._perform_security_checks(query)
            assert result.is_valid is False
            assert result.error_type == "potential_injection"


class TestCypherValidatorPatternMatching:
    """Test suite for safe pattern validation"""

    def test_validate_against_patterns_safe_match(self):
        """Test validation recognizes safe query patterns"""
        validator = CypherValidator()
        
        # These should match safe patterns
        safe_queries = [
            "MATCH (n:Context) RETURN n",
            "MATCH (n:Context) WHERE n.type = 'doc' RETURN n.id",
            "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 10"
        ]
        
        for query in safe_queries:
            result = validator._validate_against_patterns(query)
            # Should either have no warnings or minimal warnings
            assert result.is_valid is True

    def test_validate_against_patterns_no_match_warning(self):
        """Test validation warns when query doesn't match safe patterns"""
        validator = CypherValidator()
        
        # Complex query unlikely to match simple safe patterns
        query = "MATCH (n:Context) WITH n WHERE n.score > 0.5 UNWIND n.tags AS tag RETURN DISTINCT tag"
        
        result = validator._validate_against_patterns(query)
        
        assert result.is_valid is True
        # Should have warning about not matching safe patterns
        assert any("safe patterns" in warning for warning in result.warnings)


class TestCypherValidatorIntegration:
    """Integration tests for complete validation workflow"""

    def test_validate_query_complete_success(self):
        """Test complete validation workflow for successful query"""
        validator = CypherValidator()
        query = "MATCH (n:Context) WHERE n.type = $type RETURN n.id, n.title ORDER BY n.created_at DESC LIMIT 20"
        parameters = {"type": "document"}
        
        result = validator.validate_query(query, parameters)
        
        assert result.is_valid is True
        assert result.complexity_score > 0
        assert isinstance(result.warnings, list)

    def test_validate_query_with_warnings(self):
        """Test validation succeeds but includes warnings"""
        validator = CypherValidator()
        query = "MATCH (n:Context) RETURN n.*, m.*, p.* LIMIT 2000"  # Many wildcards, large limit
        
        result = validator.validate_query(query)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        # Should have warnings about large limit
        assert any("Large LIMIT value" in warning for warning in result.warnings)

    def test_validate_query_exception_handling(self):
        """Test validation handles exceptions gracefully"""
        validator = CypherValidator()
        
        # Mock _normalize_query to raise an exception
        with patch.object(validator, '_normalize_query', side_effect=Exception("Test error")):
            result = validator.validate_query("MATCH (n) RETURN n")
            
            assert result.is_valid is False
            assert result.error_type == "validation_exception"
            assert "Test error" in result.error_message

    def test_validate_query_complex_real_world(self):
        """Test validation with complex real-world query"""
        validator = CypherValidator()
        
        query = """
        // Find related contexts
        MATCH (source:Context {id: $source_id})
        MATCH (source)-[:RELATES_TO*1..2]-(related:Context)
        WHERE related.type IN ['document', 'note'] 
        AND related.created_at > $min_date
        RETURN DISTINCT related.id, related.title, related.type
        ORDER BY related.relevance_score DESC
        LIMIT 50
        """
        
        parameters = {
            "source_id": "ctx-123",
            "min_date": "2023-01-01"
        }
        
        result = validator.validate_query(query, parameters)
        
        assert result.is_valid is True
        assert result.complexity_score > 0

    def test_is_query_safe_convenience_method(self):
        """Test is_query_safe convenience method"""
        validator = CypherValidator()
        
        safe_query = "MATCH (n:Context) RETURN n LIMIT 10"
        unsafe_query = "CREATE (n:Context)"
        
        assert validator.is_query_safe(safe_query) is True
        assert validator.is_query_safe(unsafe_query) is False

    def test_get_safe_query_examples(self):
        """Test safe query examples method"""
        validator = CypherValidator()
        
        examples = validator.get_safe_query_examples()
        
        assert isinstance(examples, list)
        assert len(examples) > 0
        
        # All examples should be valid
        for example in examples:
            assert validator.is_query_safe(example) is True


class TestCypherValidatorEdgeCases:
    """Test suite for edge cases and boundary conditions"""

    def test_validate_empty_query(self):
        """Test validation of empty query"""
        validator = CypherValidator()
        
        result = validator.validate_query("")
        
        assert result.is_valid is True  # Empty query is technically safe

    def test_validate_whitespace_only_query(self):
        """Test validation of whitespace-only query"""
        validator = CypherValidator()
        
        result = validator.validate_query("   \n\t   ")
        
        assert result.is_valid is True  # Normalizes to empty, which is safe

    def test_validate_query_boundary_complexity(self):
        """Test validation at complexity boundary"""
        validator = CypherValidator(max_complexity=50)
        
        # Create query with exactly 50 complexity points (5 MATCH * 10)
        query = "MATCH (n1) MATCH (n2) MATCH (n3) MATCH (n4) MATCH (n5) RETURN n1"
        
        result = validator.validate_query(query)
        
        assert result.is_valid is True
        assert result.complexity_score == 50

    def test_validate_query_boundary_length(self):
        """Test validation at length boundary"""
        validator = CypherValidator(max_length=50)
        
        # Create query with exactly 50 characters
        query = "MATCH (n:Context) RETURN n.id LIMIT 10"  # Should be close to 50 chars
        
        if len(query) <= 50:
            result = validator.validate_query(query)
            assert result.is_valid is True
        else:
            # Adjust query to be exactly at boundary
            query = query[:50]
            result = validator.validate_query(query)
            # May or may not be valid depending on syntax, but shouldn't crash

    def test_validate_parameters_edge_cases(self):
        """Test parameter validation edge cases"""
        validator = CypherValidator()
        
        edge_case_params = [
            {},  # Empty parameters
            {"a": None},  # None value
            {"key": ""},  # Empty string
            {"k": 0},  # Zero value
            {"_": "_"},  # Minimal valid key
            {"very_long_key_name_that_is_still_valid": "value"}
        ]
        
        for params in edge_case_params:
            result = validator._validate_parameters(params)
            # Should not crash, may have warnings but should be valid
            assert isinstance(result, ValidationResult)

    def test_validate_query_unicode_content(self):
        """Test validation with unicode content"""
        validator = CypherValidator()
        
        query = "MATCH (n:Context) WHERE n.title = $title RETURN n"
        parameters = {"title": "测试内容 🔍 emoji content"}
        
        result = validator.validate_query(query, parameters)
        
        assert result.is_valid is True

    def test_validate_query_very_long_parameter_value(self):
        """Test validation with very long parameter values"""
        validator = CypherValidator()
        
        query = "MATCH (n) WHERE n.content = $content RETURN n"
        parameters = {"content": "x" * 1001}  # Just over the 1000 char threshold
        
        result = validator.validate_query(query, parameters)
        
        assert result.is_valid is True
        # Note: Parameter warnings aren't currently propagated to main result
        # This test verifies the validation doesn't fail with long parameters


class TestCypherValidatorLogging:
    """Test suite for logging behavior"""

    def test_validation_success_logging(self):
        """Test that successful validation logs appropriately"""
        validator = CypherValidator()
        query = "MATCH (n) RETURN n"
        
        with patch('src.security.cypher_validator.logger') as mock_logger:
            result = validator.validate_query(query)
            
            assert result.is_valid is True
            mock_logger.info.assert_called_once()
            assert "complexity score" in mock_logger.info.call_args[0][0]

    def test_validation_exception_logging(self):
        """Test that validation exceptions are logged"""
        validator = CypherValidator()
        
        with patch.object(validator, '_normalize_query', side_effect=Exception("Test error")):
            with patch('src.security.cypher_validator.logger') as mock_logger:
                result = validator.validate_query("MATCH (n) RETURN n")
                
                assert result.is_valid is False
                mock_logger.error.assert_called_once()
                assert "Test error" in mock_logger.error.call_args[0][0]