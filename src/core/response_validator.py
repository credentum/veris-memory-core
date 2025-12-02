#!/usr/bin/env python3
"""
Response validation framework to prevent MCP contract mismatches.

This module provides utilities to validate MCP tool responses against
their expected schemas to catch contract violations early.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error."""
    field: str
    expected: str
    actual: str
    message: str


@dataclass
class ValidationResult:
    """Result of response validation."""
    valid: bool
    errors: List[ValidationError]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.valid


class ResponseValidator:
    """Validates MCP tool responses against expected schemas."""
    
    def __init__(self):
        self.schemas = self._load_response_schemas()
    
    def _load_response_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load response schemas for validation."""
        return {
            "store_context": {
                "required_fields": {"success", "id", "message"},
                "optional_fields": {"context_id", "vector_id", "graph_id", "backend_status", "qa_enhancement", "error_type"},
                "field_types": {
                    "success": bool,
                    "id": (str, type(None)),
                    "context_id": (str, type(None)),
                    "vector_id": (str, type(None)), 
                    "graph_id": (str, type(None)),
                    "message": str,
                    "backend_status": dict,
                    "qa_enhancement": dict,
                    "error_type": str
                },
                "constraints": {
                    # When successful, id should not be null
                    "success_constraints": {
                        "when_success_true": {"id": {"not_null": True}},
                        "when_success_false": {"id": {"allow_null": True}}
                    },
                    # id and context_id should match when both present
                    "field_consistency": {
                        "id_context_id_match": ["id", "context_id"]
                    }
                }
            },
            "retrieve_context": {
                "required_fields": {"success", "results", "message"},
                "optional_fields": {"query_expansion_metadata", "retrieval_metadata", "error_type"},
                "field_types": {
                    "success": bool,
                    "results": list,
                    "message": str
                }
            },
            "query_graph": {
                "required_fields": {"success", "results"},
                "optional_fields": {"message", "error", "query_metadata"},
                "field_types": {
                    "success": bool,
                    "results": list,
                    "message": str,
                    "error": str
                }
            }
        }
    
    def validate_response(self, tool_name: str, response: Dict[str, Any]) -> ValidationResult:
        """Validate a tool response against its schema.
        
        Args:
            tool_name: Name of the MCP tool
            response: The response dict to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        if tool_name not in self.schemas:
            warnings.append(f"No schema defined for tool '{tool_name}'")
            return ValidationResult(valid=True, errors=errors, warnings=warnings)
        
        schema = self.schemas[tool_name]
        
        # Check required fields
        required_fields = schema.get("required_fields", set())
        for field in required_fields:
            if field not in response:
                errors.append(ValidationError(
                    field=field,
                    expected="present",
                    actual="missing",
                    message=f"Required field '{field}' is missing"
                ))
        
        # Check field types
        field_types = schema.get("field_types", {})
        for field, expected_type in field_types.items():
            if field in response:
                actual_value = response[field]
                if not isinstance(actual_value, expected_type):
                    errors.append(ValidationError(
                        field=field,
                        expected=str(expected_type),
                        actual=str(type(actual_value)),
                        message=f"Field '{field}' has wrong type. Expected {expected_type}, got {type(actual_value)}"
                    ))
        
        # Check constraints
        constraints = schema.get("constraints", {})
        
        # Success-based constraints
        success_constraints = constraints.get("success_constraints", {})
        if "success" in response:
            success_value = response["success"]
            
            if success_value and "when_success_true" in success_constraints:
                for field, rules in success_constraints["when_success_true"].items():
                    if field in response:
                        if rules.get("not_null") and response[field] is None:
                            errors.append(ValidationError(
                                field=field,
                                expected="non-null",
                                actual="null",
                                message=f"Field '{field}' should not be null when success=True"
                            ))
            
            if not success_value and "when_success_false" in success_constraints:
                for field, rules in success_constraints["when_success_false"].items():
                    if field in response and not rules.get("allow_null", False):
                        if response[field] is None:
                            warnings.append(f"Field '{field}' is null when success=False (may be expected)")
        
        # Field consistency constraints
        field_consistency = constraints.get("field_consistency", {})
        for constraint_name, fields in field_consistency.items():
            if constraint_name == "id_context_id_match":
                if all(field in response for field in fields):
                    id_val = response[fields[0]]
                    context_id_val = response[fields[1]]
                    if id_val is not None and context_id_val is not None and id_val != context_id_val:
                        errors.append(ValidationError(
                            field="id/context_id",
                            expected="matching values",
                            actual=f"id={id_val}, context_id={context_id_val}",
                            message="Fields 'id' and 'context_id' should have the same value"
                        ))
        
        # Check for unexpected fields (warnings only)
        expected_fields = required_fields | schema.get("optional_fields", set())
        for field in response:
            if field not in expected_fields:
                warnings.append(f"Unexpected field '{field}' in response")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_store_context_response(self, response: Dict[str, Any]) -> ValidationResult:
        """Convenience method for validating store_context responses."""
        return self.validate_response("store_context", response)
    
    def log_validation_result(self, tool_name: str, result: ValidationResult) -> None:
        """Log validation results."""
        if result.valid:
            if result.warnings:
                logger.warning(f"{tool_name} response validation passed with warnings: {result.warnings}")
            else:
                logger.debug(f"{tool_name} response validation passed")
        else:
            error_msgs = [error.message for error in result.errors]
            logger.error(f"{tool_name} response validation failed: {error_msgs}")
            if result.warnings:
                logger.warning(f"{tool_name} response validation warnings: {result.warnings}")


# Global validator instance
response_validator = ResponseValidator()


def validate_mcp_response(tool_name: str, response: Dict[str, Any], log_results: bool = True) -> ValidationResult:
    """Validate an MCP response and optionally log results.
    
    Args:
        tool_name: Name of the MCP tool
        response: Response dictionary to validate
        log_results: Whether to log validation results
        
    Returns:
        ValidationResult
    """
    result = response_validator.validate_response(tool_name, response)
    
    if log_results:
        response_validator.log_validation_result(tool_name, result)
    
    return result