#!/usr/bin/env python3
"""
v1_spec.py: Veris Memory v1.0 API Specification Validation

This module provides validation and enforcement of the v1.0 API contract
ensuring all responses conform to the standardized OpenAPI specification.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

import yaml
from pathlib import Path


class ErrorCode(str, Enum):
    """Standardized v1.0 error codes"""
    VALIDATION = "ERR_VALIDATION"
    TIMEOUT = "ERR_TIMEOUT"
    AUTH = "ERR_AUTH"
    RATE_LIMIT = "ERR_RATE_LIMIT"
    DEPENDENCY_DOWN = "ERR_DEPENDENCY_DOWN"
    DIMENSION_MISMATCH = "ERR_DIMENSION_MISMATCH"


class V1SpecValidator:
    """Validates API responses against v1.0 specification"""
    
    def __init__(self):
        self.spec_path = Path(__file__).parent.parent.parent / "docs" / "api" / "veris-memory-v1.0.yaml"
        self.spec = self._load_spec()
    
    def _load_spec(self) -> Dict[str, Any]:
        """Load OpenAPI specification"""
        try:
            with open(self.spec_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Fallback for testing environments
            return {
                "openapi": "3.0.3",
                "info": {"title": "Veris Memory API", "version": "1.0.0"}
            }
    
    def generate_trace_id(self) -> str:
        """Generate unique trace ID for request tracking"""
        return f"trace_{uuid.uuid4().hex[:12]}"
    
    def create_error_response(
        self, 
        error_code: ErrorCode, 
        message: str,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized v1.0 error response"""
        if trace_id is None:
            trace_id = self.generate_trace_id()
        
        return {
            "success": False,
            "error_code": error_code.value,
            "message": message,
            "trace_id": trace_id,
            "details": details or {}
        }
    
    def create_success_response(
        self,
        data: Dict[str, Any],
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized v1.0 success response"""
        if trace_id is None:
            trace_id = self.generate_trace_id()
        
        # Ensure success field is present
        response = {"success": True, **data}
        
        # Add trace_id to headers context (handled by middleware)
        response["_trace_id"] = trace_id
        
        return response
    
    def validate_store_context_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate store_context request against v1.0 spec"""
        errors = []
        
        # Required fields
        if "content" not in request:
            errors.append("Missing required field: content")
        if "type" not in request:
            errors.append("Missing required field: type")
        
        # Type validation
        if "type" in request:
            valid_types = ["design", "decision", "trace", "sprint", "log"]
            if request["type"] not in valid_types:
                errors.append(f"Invalid type: must be one of {valid_types}")
        
        # Metadata validation
        if "metadata" in request and "priority" in request["metadata"]:
            valid_priorities = ["low", "medium", "high", "critical"]
            if request["metadata"]["priority"] not in valid_priorities:
                errors.append(f"Invalid priority: must be one of {valid_priorities}")
        
        if errors:
            return self.create_error_response(
                ErrorCode.VALIDATION,
                "Input validation failed",
                details={"validation_errors": errors}
            )
        
        return None
    
    def validate_retrieve_context_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate retrieve_context request against v1.0 spec"""
        errors = []
        
        # Required fields
        if "query" not in request:
            errors.append("Missing required field: query")
        
        # Query length validation
        if "query" in request and len(request["query"]) > 50000:
            errors.append("Query too long: maximum 50,000 characters")
        
        # Type validation
        if "type" in request:
            valid_types = ["design", "decision", "trace", "sprint", "log", "all"]
            if request["type"] not in valid_types:
                errors.append(f"Invalid type: must be one of {valid_types}")
        
        # Search mode validation
        if "search_mode" in request:
            valid_modes = ["vector", "graph", "hybrid"]
            if request["search_mode"] not in valid_modes:
                errors.append(f"Invalid search_mode: must be one of {valid_modes}")
        
        # Limit validation
        if "limit" in request:
            if not isinstance(request["limit"], int) or request["limit"] < 1 or request["limit"] > 100:
                errors.append("Invalid limit: must be integer between 1 and 100")
        
        # Sort validation
        if "sort_by" in request:
            valid_sorts = ["timestamp", "relevance"]
            if request["sort_by"] not in valid_sorts:
                errors.append(f"Invalid sort_by: must be one of {valid_sorts}")
        
        if errors:
            return self.create_error_response(
                ErrorCode.VALIDATION,
                "Input validation failed",
                details={"validation_errors": errors}
            )
        
        return None
    
    def validate_query_graph_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate query_graph request against v1.0 spec"""
        errors = []
        
        # Required fields
        if "query" not in request:
            errors.append("Missing required field: query")
        
        # Query length validation
        if "query" in request and len(request["query"]) > 10000:
            errors.append("Query too long: maximum 10,000 characters")
        
        # Basic Cypher safety validation
        if "query" in request:
            query = request["query"].upper()
            forbidden_ops = ["CREATE", "DELETE", "SET", "REMOVE", "MERGE", "DROP", "DETACH"]
            for op in forbidden_ops:
                if op in query:
                    errors.append(f"Forbidden Cypher operation: {op}")
                    break
        
        if errors:
            return self.create_error_response(
                ErrorCode.VALIDATION,
                "Input validation failed",
                details={"validation_errors": errors}
            )
        
        return None
    
    def add_rate_limit_headers(self, response: Dict[str, Any], limit: int, remaining: int) -> Dict[str, Any]:
        """Add v1.0 standard rate limiting headers"""
        if "_headers" not in response:
            response["_headers"] = {}
        
        response["_headers"]["X-RateLimit-Limit"] = str(limit)
        response["_headers"]["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def add_trace_header(self, response: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """Add trace ID header for debugging"""
        if "_headers" not in response:
            response["_headers"] = {}
        
        response["_headers"]["X-Trace-ID"] = trace_id
        
        return response


class V1ResponseFormatter:
    """Formats responses according to v1.0 specification"""
    
    def __init__(self):
        self.validator = V1SpecValidator()
    
    def format_retrieve_context_response(
        self,
        results: List[Dict[str, Any]],
        total_count: int,
        search_mode_used: str,
        limit: int,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format retrieve_context response per v1.0 spec"""
        has_more = total_count > len(results)
        
        response = self.validator.create_success_response({
            "results": results,
            "total_count": total_count,
            "has_more": has_more,
            "search_mode_used": search_mode_used,
            "message": f"Found {len(results)} matching contexts"
        }, trace_id)
        
        return response
    
    def format_store_context_response(
        self,
        context_id: str,
        vector_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format store_context response per v1.0 spec"""
        response_data = {
            "id": context_id,
            "message": "Context stored successfully"
        }
        
        if vector_id:
            response_data["vector_id"] = vector_id
        if graph_id:
            response_data["graph_id"] = graph_id
        
        return self.validator.create_success_response(response_data, trace_id)
    
    def format_query_graph_response(
        self,
        results: List[Dict[str, Any]],
        query_time_ms: float,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format query_graph response per v1.0 spec"""
        return self.validator.create_success_response({
            "results": results,
            "query_time_ms": query_time_ms
        }, trace_id)


# Export main classes for use in server
__all__ = [
    "ErrorCode",
    "V1SpecValidator", 
    "V1ResponseFormatter"
]