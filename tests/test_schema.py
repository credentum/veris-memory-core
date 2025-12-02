#!/usr/bin/env python3
"""
Schema validation tests for data ingestion pipeline.
Tests input validation, type checking, and schema compliance.
"""

import pytest
import json
from typing import Dict, Any, List
from datetime import datetime
from jsonschema import validate, ValidationError, Draft7Validator
import numpy as np


# Ingestion schemas
CONTEXT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["type", "content"],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["design", "decision", "code", "test", "documentation", "discussion"]
        },
        "content": {
            "type": "object",
            "required": ["title"],
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200
                },
                "description": {
                    "type": "string",
                    "maxLength": 5000
                },
                "body": {
                    "type": "string",
                    "maxLength": 50000
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^[a-z0-9-]+$"
                    },
                    "maxItems": 20
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "author": {"type": "string"},
                "timestamp": {
                    "type": "string",
                    "format": "date-time"
                },
                "source": {"type": "string"},
                "version": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"]
                }
            }
        },
        "embedding": {
            "type": "object",
            "properties": {
                "vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 384,
                    "maxItems": 384
                },
                "model": {"type": "string"},
                "dimensions": {
                    "type": "integer",
                    "const": 384
                }
            }
        }
    }
}

QUERY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1000
        },
        "filters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["design", "decision", "code", "test", "documentation", "discussion"]
                    }
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "date_from": {
                    "type": "string",
                    "format": "date-time"
                },
                "date_to": {
                    "type": "string",
                    "format": "date-time"
                },
                "author": {"type": "string"}
            }
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "default": 10
        },
        "threshold": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.5
        }
    }
}


class SchemaValidator:
    """Validator for ingestion schemas."""
    
    def __init__(self):
        self.context_validator = Draft7Validator(CONTEXT_SCHEMA)
        self.query_validator = Draft7Validator(QUERY_SCHEMA)
        
    def validate_context(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate context data against schema.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        for error in self.context_validator.iter_errors(data):
            errors.append(f"{'.'.join(str(p) for p in error.path)}: {error.message}")
        return errors
    
    def validate_query(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate query data against schema.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        for error in self.query_validator.iter_errors(data):
            errors.append(f"{'.'.join(str(p) for p in error.path)}: {error.message}")
        return errors
    
    def validate_embedding_dimensions(self, vector: List[float]) -> bool:
        """Validate embedding vector dimensions."""
        return len(vector) == 384
    
    def validate_embedding_values(self, vector: List[float]) -> bool:
        """Validate embedding vector values are normalized."""
        norm = np.linalg.norm(vector)
        return 0.9 <= norm <= 1.1  # Allow small tolerance


class TestContextSchema:
    """Tests for context ingestion schema."""
    
    def setup_method(self):
        self.validator = SchemaValidator()
    
    def test_valid_minimal_context(self):
        """Test minimal valid context."""
        context = {
            "type": "design",
            "content": {
                "title": "API Design"
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) == 0
    
    def test_valid_full_context(self):
        """Test fully populated context."""
        context = {
            "type": "decision",
            "content": {
                "title": "Use PostgreSQL for main database",
                "description": "Decision to use PostgreSQL",
                "body": "After evaluating options...",
                "tags": ["database", "architecture", "postgresql"]
            },
            "metadata": {
                "author": "john.doe",
                "timestamp": "2024-01-15T10:30:00Z",
                "source": "ADR-001",
                "version": "1.0.0",
                "priority": "high"
            },
            "embedding": {
                "vector": [0.1] * 384,
                "model": "all-MiniLM-L6-v2",
                "dimensions": 384
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) == 0
    
    def test_invalid_type(self):
        """Test invalid context type."""
        context = {
            "type": "invalid_type",
            "content": {
                "title": "Test"
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        assert any("type" in error for error in errors)
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        context = {
            "type": "design"
            # Missing content
        }
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        assert any("content" in error for error in errors)
    
    def test_title_validation(self):
        """Test title field validation."""
        # Empty title
        context = {
            "type": "design",
            "content": {
                "title": ""
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        
        # Title too long
        context = {
            "type": "design",
            "content": {
                "title": "x" * 201
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
    
    def test_tag_validation(self):
        """Test tag field validation."""
        # Invalid tag format
        context = {
            "type": "design",
            "content": {
                "title": "Test",
                "tags": ["valid-tag", "Invalid Tag", "also_invalid"]
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        assert any("tags" in error for error in errors)
        
        # Too many tags
        context = {
            "type": "design",
            "content": {
                "title": "Test",
                "tags": [f"tag-{i}" for i in range(21)]
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
    
    def test_embedding_validation(self):
        """Test embedding field validation."""
        # Wrong dimensions
        context = {
            "type": "design",
            "content": {
                "title": "Test"
            },
            "embedding": {
                "vector": [0.1] * 1536,  # Wrong size
                "model": "text-embedding-ada-002",
                "dimensions": 1536
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        
        # Correct dimensions
        context["embedding"]["vector"] = [0.1] * 384
        context["embedding"]["dimensions"] = 384
        errors = self.validator.validate_context(context)
        assert len(errors) == 0
    
    def test_metadata_validation(self):
        """Test metadata field validation."""
        # Invalid timestamp format
        context = {
            "type": "design",
            "content": {
                "title": "Test"
            },
            "metadata": {
                "timestamp": "2024-01-15"  # Missing time component
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        
        # Invalid priority
        context = {
            "type": "design",
            "content": {
                "title": "Test"
            },
            "metadata": {
                "priority": "urgent"  # Not in enum
            }
        }
        errors = self.validator.validate_context(context)
        assert len(errors) > 0


class TestQuerySchema:
    """Tests for query schema validation."""
    
    def setup_method(self):
        self.validator = SchemaValidator()
    
    def test_valid_minimal_query(self):
        """Test minimal valid query."""
        query = {
            "query": "microservices architecture"
        }
        errors = self.validator.validate_query(query)
        assert len(errors) == 0
    
    def test_valid_full_query(self):
        """Test fully populated query."""
        query = {
            "query": "database design patterns",
            "filters": {
                "type": ["design", "decision"],
                "tags": ["database", "architecture"],
                "date_from": "2024-01-01T00:00:00Z",
                "date_to": "2024-12-31T23:59:59Z",
                "author": "john.doe"
            },
            "limit": 20,
            "threshold": 0.7
        }
        errors = self.validator.validate_query(query)
        assert len(errors) == 0
    
    def test_empty_query(self):
        """Test empty query string."""
        query = {
            "query": ""
        }
        errors = self.validator.validate_query(query)
        assert len(errors) > 0
    
    def test_query_too_long(self):
        """Test query exceeding max length."""
        query = {
            "query": "x" * 1001
        }
        errors = self.validator.validate_query(query)
        assert len(errors) > 0
    
    def test_invalid_limit(self):
        """Test invalid limit values."""
        # Negative limit
        query = {
            "query": "test",
            "limit": -1
        }
        errors = self.validator.validate_query(query)
        assert len(errors) > 0
        
        # Limit too high
        query = {
            "query": "test",
            "limit": 101
        }
        errors = self.validator.validate_query(query)
        assert len(errors) > 0
    
    def test_invalid_threshold(self):
        """Test invalid threshold values."""
        # Threshold too low
        query = {
            "query": "test",
            "threshold": -0.1
        }
        errors = self.validator.validate_query(query)
        assert len(errors) > 0
        
        # Threshold too high
        query = {
            "query": "test",
            "threshold": 1.1
        }
        errors = self.validator.validate_query(query)
        assert len(errors) > 0
    
    def test_invalid_filter_types(self):
        """Test invalid filter type values."""
        query = {
            "query": "test",
            "filters": {
                "type": ["design", "invalid_type"]
            }
        }
        errors = self.validator.validate_query(query)
        assert len(errors) > 0


class TestEmbeddingValidation:
    """Tests for embedding-specific validation."""
    
    def setup_method(self):
        self.validator = SchemaValidator()
    
    def test_correct_dimensions(self):
        """Test embedding with correct dimensions."""
        vector = [0.1] * 384
        assert self.validator.validate_embedding_dimensions(vector) == True
    
    def test_incorrect_dimensions(self):
        """Test embedding with incorrect dimensions."""
        # Too few dimensions
        vector = [0.1] * 383
        assert self.validator.validate_embedding_dimensions(vector) == False
        
        # Too many dimensions
        vector = [0.1] * 385
        assert self.validator.validate_embedding_dimensions(vector) == False
        
        # Wrong model dimensions (1536)
        vector = [0.1] * 1536
        assert self.validator.validate_embedding_dimensions(vector) == False
    
    def test_normalized_vector(self):
        """Test normalized embedding vector."""
        # Create unit vector
        vector = [0.0] * 384
        vector[0] = 1.0
        assert self.validator.validate_embedding_values(vector) == True
        
        # Create normalized vector
        import math
        dim = 384
        value = 1.0 / math.sqrt(dim)
        vector = [value] * dim
        assert self.validator.validate_embedding_values(vector) == True
    
    def test_unnormalized_vector(self):
        """Test unnormalized embedding vector."""
        # Zero vector
        vector = [0.0] * 384
        assert self.validator.validate_embedding_values(vector) == False
        
        # Large magnitude vector
        vector = [1.0] * 384
        assert self.validator.validate_embedding_values(vector) == False


class TestSchemaEvolution:
    """Tests for schema evolution and backward compatibility."""
    
    def setup_method(self):
        self.validator = SchemaValidator()
    
    def test_additional_properties_allowed(self):
        """Test that additional properties don't break validation."""
        context = {
            "type": "design",
            "content": {
                "title": "Test",
                "future_field": "some value"  # Unknown field
            },
            "experimental": {  # Unknown top-level field
                "feature": "test"
            }
        }
        errors = self.validator.validate_context(context)
        # Should still be valid with extra fields
        assert len(errors) == 0
    
    def test_optional_fields_omitted(self):
        """Test that optional fields can be omitted."""
        context = {
            "type": "design",
            "content": {
                "title": "Test"
                # No description, body, or tags
            }
            # No metadata or embedding
        }
        errors = self.validator.validate_context(context)
        assert len(errors) == 0


@pytest.mark.parametrize("dimension_count,expected", [
    (384, True),   # Correct
    (1536, False), # OpenAI dimensions
    (768, False),  # BERT large dimensions
    (512, False),  # Other common dimension
    (0, False),    # Empty
])
def test_dimension_validation_parametrized(dimension_count, expected):
    """Parametrized test for dimension validation."""
    validator = SchemaValidator()
    vector = [0.1] * dimension_count if dimension_count > 0 else []
    assert validator.validate_embedding_dimensions(vector) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])