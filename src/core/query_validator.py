#!/usr/bin/env python3
"""
Secure Cypher query validation for Neo4j operations.
Prevents injection attacks and enforces read-only operations.
"""

import re
from enum import Enum
from typing import List, Optional, Tuple


class CypherOperation(Enum):
    """Cypher query operation types."""

    # Read operations
    MATCH = "MATCH"
    WITH = "WITH"
    RETURN = "RETURN"
    WHERE = "WHERE"
    ORDER_BY = "ORDER BY"
    LIMIT = "LIMIT"
    SKIP = "SKIP"
    UNWIND = "UNWIND"
    OPTIONAL_MATCH = "OPTIONAL MATCH"
    CALL = "CALL"

    # Write operations (forbidden)
    CREATE = "CREATE"
    DELETE = "DELETE"
    DETACH_DELETE = "DETACH DELETE"
    SET = "SET"
    REMOVE = "REMOVE"
    MERGE = "MERGE"
    DROP = "DROP"
    FOREACH = "FOREACH"


class QueryValidator:
    """Validates Cypher queries for security and safety."""

    # Patterns for detecting operations
    OPERATION_PATTERNS = {
        CypherOperation.CREATE: r"\b(CREATE)\b",
        CypherOperation.DELETE: r"\b(DELETE|DETACH\s+DELETE)\b",
        CypherOperation.SET: r"\b(SET)\b",
        CypherOperation.REMOVE: r"\b(REMOVE)\b",
        CypherOperation.MERGE: r"\b(MERGE)\b",
        CypherOperation.DROP: r"\b(DROP)\b",
        CypherOperation.FOREACH: r"\b(FOREACH)\b",
    }

    # Allowed read operations
    ALLOWED_OPERATIONS = {
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

    # Forbidden operations
    FORBIDDEN_OPERATIONS = {
        CypherOperation.CREATE,
        CypherOperation.DELETE,
        CypherOperation.DETACH_DELETE,
        CypherOperation.SET,
        CypherOperation.REMOVE,
        CypherOperation.MERGE,
        CypherOperation.DROP,
        CypherOperation.FOREACH,
    }

    # Patterns that might indicate injection attempts
    INJECTION_PATTERNS = [
        r";\s*(CREATE|DELETE|SET|REMOVE|MERGE|DROP)",  # Command chaining
        r"/\*.*?\*/",  # Block comments that might hide operations
        r"--.*$",  # Line comments that might hide operations
        r"\\x[0-9a-fA-F]{2}",  # Hex escapes
        r"\\[0-7]{3}",  # Octal escapes
        r"char\s*\(",  # Character construction
        r"concat\s*\(",  # String concatenation functions
    ]

    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, Optional[str]]:
        """Validate a Cypher query for safety.

        Args:
            query: The Cypher query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"

        # Check query length
        if len(query) > 10000:
            return False, "Query exceeds maximum length"

        # Remove string literals to avoid false positives
        cleaned_query = cls._remove_string_literals(query)

        # Check for forbidden operations
        for operation, pattern in cls.OPERATION_PATTERNS.items():
            if re.search(pattern, cleaned_query, re.IGNORECASE | re.MULTILINE):
                return False, f"Forbidden operation detected: {operation.value}"

        # Check for injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE | re.MULTILINE):
                return False, "Potential injection pattern detected"

        # Check for command chaining
        if cls._has_command_chaining(cleaned_query):
            return False, "Command chaining is not allowed"

        # Validate parentheses balance
        if not cls._validate_parentheses(query):
            return False, "Unbalanced parentheses detected"

        # Validate quotes balance
        if not cls._validate_quotes(query):
            return False, "Unbalanced quotes detected"

        return True, None

    @staticmethod
    def _remove_string_literals(query: str) -> str:
        """Remove string literals from query to avoid false positives.

        Args:
            query: The query to clean

        Returns:
            Query with string literals replaced
        """
        # Replace single-quoted strings
        query = re.sub(r"'[^']*'", "''", query)
        # Replace double-quoted strings
        query = re.sub(r'"[^"]*"', '""', query)
        # Replace backtick-quoted identifiers
        query = re.sub(r"`[^`]*`", "``", query)
        return query

    @staticmethod
    def _has_command_chaining(query: str) -> bool:
        """Check if query attempts to chain multiple commands.

        Args:
            query: The query to check

        Returns:
            True if command chaining is detected
        """
        # Look for semicolons not in strings
        # Simple check - could be enhanced
        semicolons = re.findall(r";", query)
        return len(semicolons) > 0

    @staticmethod
    def _validate_parentheses(query: str) -> bool:
        """Validate that parentheses are balanced.

        Args:
            query: The query to validate

        Returns:
            True if parentheses are balanced
        """
        count = 0
        for char in query:
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
                if count < 0:
                    return False
        return count == 0

    @staticmethod
    def _validate_quotes(query: str) -> bool:
        """Validate that quotes are balanced.

        Args:
            query: The query to validate

        Returns:
            True if quotes are balanced
        """
        single_quotes = query.count("'") - query.count("\\'")
        double_quotes = query.count('"') - query.count('\\"')
        backticks = query.count("`")

        return single_quotes % 2 == 0 and double_quotes % 2 == 0 and backticks % 2 == 0

    @classmethod
    def sanitize_parameters(cls, params: dict) -> dict:
        """Sanitize query parameters to prevent injection.

        Args:
            params: Query parameters

        Returns:
            Sanitized parameters
        """
        sanitized = {}
        for key, value in params.items():
            # Validate parameter names
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(f"Invalid parameter name: {key}")

            # Sanitize values
            if isinstance(value, str):
                # Remove any control characters
                value = re.sub(r"[\x00-\x1f\x7f]", "", value)

            sanitized[key] = value

        return sanitized

    @classmethod
    def create_safe_query(
        cls,
        pattern: str,
        filters: Optional[dict] = None,
        return_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> str:
        """Create a safe read-only query.

        Args:
            pattern: The MATCH pattern
            filters: WHERE clause filters
            return_fields: Fields to return
            limit: Result limit

        Returns:
            Safe Cypher query string
        """
        # Start with MATCH
        query_parts = [f"MATCH {pattern}"]

        # Add WHERE clause if filters provided
        if filters:
            where_conditions = []
            for key, value in filters.items():
                # Validate key
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", key):
                    raise ValueError(f"Invalid filter key: {key}")

                if isinstance(value, str):
                    where_conditions.append(f"{key} = '{cls._escape_string(value)}'")
                elif isinstance(value, bool):
                    where_conditions.append(f"{key} = {str(value).lower()}")
                elif isinstance(value, (int, float)):
                    where_conditions.append(f"{key} = {value}")
                elif value is None:
                    where_conditions.append(f"{key} IS NULL")

            if where_conditions:
                query_parts.append(f"WHERE {' AND '.join(where_conditions)}")

        # Add RETURN clause
        if return_fields:
            # Validate return fields
            validated_fields = []
            for field in return_fields:
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", field):
                    raise ValueError(f"Invalid return field: {field}")
                validated_fields.append(field)
            query_parts.append(f"RETURN {', '.join(validated_fields)}")
        else:
            query_parts.append("RETURN *")

        # Add LIMIT if specified
        if limit:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError("Limit must be a positive integer")
            query_parts.append(f"LIMIT {limit}")

        return " ".join(query_parts)

    @staticmethod
    def _escape_string(value: str) -> str:
        """Escape string value for safe inclusion in query.

        Args:
            value: String to escape

        Returns:
            Escaped string
        """
        # Escape single quotes
        value = value.replace("'", "\\'")
        # Escape backslashes
        value = value.replace("\\", "\\\\")
        # Remove control characters
        value = re.sub(r"[\x00-\x1f\x7f]", "", value)
        return value


def validate_cypher_query(query: str) -> Tuple[bool, Optional[str]]:
    """Convenience function to validate a Cypher query.

    Args:
        query: The query to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    return QueryValidator.validate_query(query)
