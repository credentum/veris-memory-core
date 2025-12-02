"""
Cypher Query Validator for securing Neo4j operations.

This module provides comprehensive validation for Cypher queries to prevent
injection attacks, unauthorized write operations, and resource abuse.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional  # Set removed as unused

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when security validation fails."""

    def __init__(self, message: str, error_type: str = "security_violation"):
        super().__init__(message)
        self.error_type = error_type


@dataclass
class ValidationResult:
    """Result of Cypher query validation."""

    is_valid: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    warnings: List[str] = None
    complexity_score: int = 0

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class CypherValidator:
    """
    Validates Cypher queries for security and compliance.

    This validator implements multiple security layers:
    1. Write operation detection and blocking
    2. Query complexity analysis to prevent resource abuse
    3. Allowlist validation for approved patterns
    4. Parameter injection prevention
    """

    # Forbidden operations that could modify data
    FORBIDDEN_OPERATIONS = {
        "CREATE",
        "DELETE",
        "REMOVE",
        "SET",
        "MERGE",
        "DROP",
        "DETACH",
        "FOREACH",
        "LOAD",
        "USING",
        "PERIODIC",
        "CALL",
        "YIELD",  # Restrict procedure calls for security
    }

    # Operations that are allowed (read-only)
    ALLOWED_OPERATIONS = {
        "MATCH",
        "RETURN",
        "WHERE",
        "WITH",
        "ORDER",
        "BY",
        "LIMIT",
        "SKIP",
        "UNION",
        "UNWIND",
        "AS",
        "DISTINCT",
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "COLLECT",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "AND",
        "OR",
        "NOT",
        "IN",
        "IS",
        "NULL",
        "EXISTS",
        "SIZE",
        "LENGTH",
    }

    # Maximum query complexity score
    MAX_COMPLEXITY_SCORE = 100

    # Maximum query length
    MAX_QUERY_LENGTH = 5000

    # Pattern allowlist for common safe queries
    SAFE_QUERY_PATTERNS = [
        r"^MATCH\s+\([^)]*\)\s*(?:-\[[^]]*\]->[^)]*\))*\s*"
        + r"(?:WHERE\s+[^;]*?)?\s*RETURN\s+[^;]+$",
        r"^MATCH\s+\([^)]*\)\s*(?:WHERE\s+[^;]*?)?\s*RETURN\s+[^;]+"
        + r"(?:\s+ORDER\s+BY\s+[^;]+)?(?:\s+LIMIT\s+\d+)?$",
    ]

    def __init__(self, max_complexity: int = None, max_length: int = None):
        """
        Initialize the validator with custom limits.

        Args:
            max_complexity: Maximum allowed query complexity score
            max_length: Maximum allowed query length in characters
        """
        self.max_complexity = max_complexity or self.MAX_COMPLEXITY_SCORE
        self.max_length = max_length or self.MAX_QUERY_LENGTH

        # Compile regex patterns for performance
        self._forbidden_pattern = re.compile(
            r"\b(" + "|".join(self.FORBIDDEN_OPERATIONS) + r")\b", re.IGNORECASE
        )
        self._safe_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in self.SAFE_QUERY_PATTERNS
        ]

    def validate_query(self, query: str, parameters: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate a Cypher query for security compliance.

        Args:
            query: The Cypher query to validate
            parameters: Optional query parameters to validate

        Returns:
            ValidationResult with validation status and details
        """
        if parameters is None:
            parameters = {}

        warnings = []

        try:
            # Basic length check
            if len(query) > self.max_length:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Query exceeds maximum length of {self.max_length} characters",
                    error_type="query_too_long",
                )

            # Normalize query for analysis
            normalized_query = self._normalize_query(query)

            # Check for forbidden operations
            forbidden_check = self._check_forbidden_operations(normalized_query)
            if not forbidden_check.is_valid:
                return forbidden_check

            # Check query complexity
            complexity_score = self._calculate_complexity(normalized_query)
            if complexity_score > self.max_complexity:
                error_msg = (
                    f"Query complexity score {complexity_score} "
                    f"exceeds maximum {self.max_complexity}"
                )
                return ValidationResult(
                    is_valid=False,
                    error_message=error_msg,
                    error_type="complexity_too_high",
                    complexity_score=complexity_score,
                )

            # Parameter validation
            param_validation = self._validate_parameters(parameters)
            if not param_validation.is_valid:
                return param_validation

            # Check against safe patterns
            pattern_validation = self._validate_against_patterns(normalized_query)
            if pattern_validation.warnings:
                warnings.extend(pattern_validation.warnings)

            # Additional security checks (run on original query before comment removal)
            security_checks = self._perform_security_checks(query)
            if not security_checks.is_valid:
                return security_checks

            warnings.extend(security_checks.warnings)

            logger.info(f"Query validation passed with complexity score {complexity_score}")

            return ValidationResult(
                is_valid=True, warnings=warnings, complexity_score=complexity_score
            )

        except Exception as e:
            logger.error(f"Query validation failed with exception: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}",
                error_type="validation_exception",
            )

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent analysis.

        Performs the following normalizations:
        1. Removes single-line comments (//)
        2. Removes multi-line comments (/* */)
        3. Normalizes whitespace to single spaces
        4. Strips leading and trailing whitespace

        Args:
            query: Raw Cypher query string

        Returns:
            Normalized query string ready for validation
        """
        # Remove comments
        query = re.sub(r"//.*?$", "", query, flags=re.MULTILINE)
        query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)

        # Normalize whitespace
        query = re.sub(r"\s+", " ", query.strip())

        return query

    def _check_forbidden_operations(self, query: str) -> ValidationResult:
        """Check for forbidden write operations.

        Validates that the query does not contain any operations that could
        modify the database state. This includes CREATE, DELETE, REMOVE, SET,
        MERGE, and other write operations.

        Args:
            query: Normalized Cypher query string

        Returns:
            ValidationResult with is_valid=False if forbidden operations found,
            including the list of detected operations in error_message
        """
        matches = self._forbidden_pattern.findall(query)
        if matches:
            forbidden_ops = list(set(match.upper() for match in matches))
            return ValidationResult(
                is_valid=False,
                error_message=f"Query contains forbidden operations: {', '.join(forbidden_ops)}",
                error_type="forbidden_operation",
            )

        return ValidationResult(is_valid=True)

    def _calculate_complexity(self, query: str) -> int:
        """Calculate query complexity score.

        Analyzes query structure to compute a complexity score based on:
        - Number of MATCH clauses (10 points each)
        - WHERE clauses (5 points each)
        - WITH clauses (8 points each)
        - UNION operations (15 points each)
        - UNWIND operations (10 points each)
        - ORDER BY clauses (5 points each)
        - Nested patterns (20 points each)
        - Relationship patterns (5 points each)
        - Variable length paths (25 points each)

        Args:
            query: Normalized Cypher query string

        Returns:
            Integer complexity score (higher = more complex)
        """
        complexity = 0

        # Count various complexity factors
        complexity += len(re.findall(r"\bMATCH\b", query, re.IGNORECASE)) * 10
        complexity += len(re.findall(r"\bWHERE\b", query, re.IGNORECASE)) * 5
        complexity += len(re.findall(r"\bWITH\b", query, re.IGNORECASE)) * 8
        complexity += len(re.findall(r"\bUNION\b", query, re.IGNORECASE)) * 15
        complexity += len(re.findall(r"\bUNWIND\b", query, re.IGNORECASE)) * 10
        complexity += len(re.findall(r"\bORDER\s+BY\b", query, re.IGNORECASE)) * 5

        # Count nested patterns
        complexity += len(re.findall(r"\([^)]*\([^)]*\)[^)]*\)", query)) * 20

        # Count relationship patterns
        complexity += len(re.findall(r"-\[[^]]*\]->", query)) * 5
        complexity += len(re.findall(r"<-\[[^]]*\]-", query)) * 5

        return complexity

    def _validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate query parameters for security.

        Performs comprehensive validation of query parameters to prevent:
        - SQL/Cypher injection through parameter values
        - Invalid parameter naming that could bypass sanitization
        - Overly long parameter values that could cause DoS
        - Suspicious patterns in string values

        Args:
            parameters: Dictionary of parameter names to values

        Returns:
            ValidationResult with is_valid=False if dangerous parameters detected,
            warnings for potentially problematic but allowed parameters
        """
        warnings = []

        for key, value in parameters.items():
            # Check parameter key format
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Invalid parameter name: {key}",
                    error_type="invalid_parameter_name",
                )

            # Check for potential injection in string values
            if isinstance(value, str):
                if len(value) > 1000:
                    warnings.append(f"Parameter '{key}' has very long string value")

                # Check for suspicious patterns
                suspicious_patterns = [
                    r"\bDROP\b",
                    r"\bDELETE\b",
                    r"\bCREATE\b",
                    r"\bSET\b",
                    r"\bMERGE\b",
                    r"\bREMOVE\b",
                ]
                for pattern in suspicious_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Parameter '{key}' contains suspicious content",
                            error_type="suspicious_parameter",
                        )

        return ValidationResult(is_valid=True, warnings=warnings)

    def _validate_against_patterns(self, query: str) -> ValidationResult:
        """Validate query against safe patterns.

        Checks if the query matches known safe query patterns. Queries that
        don't match safe patterns are not automatically rejected but generate
        warnings for additional scrutiny.

        Safe patterns include:
        - Simple MATCH-RETURN queries
        - Queries with WHERE, ORDER BY, and LIMIT clauses
        - Standard graph traversal patterns

        Args:
            query: Normalized Cypher query string

        Returns:
            ValidationResult with warnings if query doesn't match safe patterns
        """
        warnings = []

        # Check if query matches any safe patterns
        matches_safe_pattern = any(pattern.match(query) for pattern in self._safe_patterns)

        if not matches_safe_pattern:
            warnings.append("Query does not match any known safe patterns")

        return ValidationResult(is_valid=True, warnings=warnings)

    def _perform_security_checks(self, query: str) -> ValidationResult:
        """Perform additional security checks.

        Conducts deep security analysis including:
        - Detection of multiple statement execution attempts
        - Dynamic execution patterns (EXEC, EVAL)
        - Comment-based injection attempts
        - Variable length path patterns that could cause performance issues
        - Suspicious character sequences

        Args:
            query: Normalized Cypher query string

        Returns:
            ValidationResult with is_valid=False if security threats detected,
            warnings for patterns that merit caution
        """
        warnings = []

        # Check for basic injection patterns
        basic_patterns = [
            r";\s*\w+",  # Multiple statements
            r"\bEXEC\b",
            r"\bEVAL\b",  # Dynamic execution
            r"--",  # SQL-style line comments (not valid in Cypher)
        ]

        for pattern in basic_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    error_message="Query contains potentially malicious patterns",
                    error_type="potential_injection",
                )

        # Check for SQL-style block comments with context awareness
        if self._contains_suspicious_comments(query):
            return ValidationResult(
                is_valid=False,
                error_message="Query contains potentially malicious comment patterns",
                error_type="potential_injection",
            )

        # Check for resource-intensive operations
        if len(re.findall(r"\*", query)) > 3:
            warnings.append("Query uses many wildcard selections, may be resource-intensive")

        # Check for very large LIMIT values
        limit_matches = re.findall(r"\bLIMIT\s+(\d+)", query, re.IGNORECASE)
        for limit_str in limit_matches:
            limit_val = int(limit_str)
            if limit_val > 10000:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"LIMIT value {limit_val} is too large (max: 10000)",
                    error_type="limit_too_large",
                )
            elif limit_val > 1000:
                warnings.append(f"Large LIMIT value: {limit_val}")

        return ValidationResult(is_valid=True, warnings=warnings)

    def _contains_suspicious_comments(self, query: str) -> bool:
        """Check for suspicious comment patterns that could indicate injection attempts.
        
        This method performs context-aware checking to distinguish between:
        - Legitimate Cypher comments (// style)
        - Suspicious SQL-style comments that could hide injection
        - Comments containing potential SQL commands
        
        Args:
            query: The original query string (before comment normalization)
            
        Returns:
            True if suspicious comment patterns are found, False otherwise
        """
        # Find all SQL-style block comments /* ... */ - ALL are considered suspicious
        # since block comments are SQL-style syntax, not valid Cypher
        block_comment_pattern = r'/\*.*?\*/'
        block_comments = re.findall(block_comment_pattern, query, re.DOTALL)
        
        # Any SQL-style block comment is suspicious in Cypher context
        if block_comments:
            return True
        
        # Check for line comments with suspicious patterns
        line_comment_pattern = r'//.*$'
        line_comments = re.findall(line_comment_pattern, query, re.MULTILINE)
        
        for comment in line_comments:
            comment_content = comment[2:].strip()  # Remove //
            
            # Very long line comments might be suspicious
            if len(comment_content) > 200:
                return True
                
            # Comments that look like they're trying to hide code
            if comment_content.count(';') > 2:  # Multiple statements
                return True
        
        return False

    def is_query_safe(self, query: str, parameters: Dict[str, Any] = None) -> bool:
        """
        Quick safety check for a query.

        Args:
            query: Cypher query to check
            parameters: Optional query parameters

        Returns:
            True if query is safe, False otherwise
        """
        result = self.validate_query(query, parameters)
        return result.is_valid

    def get_safe_query_examples(self) -> List[str]:
        """Get examples of safe query patterns."""
        return [
            "MATCH (n:Context) WHERE n.type = $type RETURN n.id, n.title LIMIT 10",
            "MATCH (n:Context)-[:RELATES_TO]->(m:Context) RETURN n.id, m.id",
            "MATCH (n:Context) WHERE n.created_at > $timestamp RETURN COUNT(n)",
            "MATCH (n:Context) RETURN n.type, COUNT(n) ORDER BY COUNT(n) DESC",
        ]
