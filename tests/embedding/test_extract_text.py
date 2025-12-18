"""
Tests for recursive text extraction in embedding service.

These tests verify that nested content is properly extracted for embedding,
preventing the "lobotomy" issue where rich nested content was lost.
"""

import pytest
from src.embedding.service import EmbeddingService


@pytest.fixture
def embedding_service():
    """Create an EmbeddingService instance for testing."""
    # Use a mock config that doesn't require actual model loading
    return EmbeddingService.__new__(EmbeddingService)


class TestExtractText:
    """Tests for _extract_text method."""

    def test_extract_string(self, embedding_service):
        """Test that plain strings are returned as-is."""
        result = embedding_service._extract_text("hello world")
        assert result == "hello world"

    def test_extract_simple_dict(self, embedding_service):
        """Test extraction from simple dict with title."""
        content = {"title": "Test Title", "description": "Test description"}
        result = embedding_service._extract_text(content)

        assert "TITLE: Test Title" in result
        assert "Test description" in result

    def test_extract_nested_dict(self, embedding_service):
        """Test extraction from nested dict structure."""
        content = {
            "title": "Phase 3 Features",
            "features": {
                "precedent_storage": {
                    "endpoint": "POST /learning/precedents/store",
                    "description": "Store learnings from completed tasks"
                },
                "precedent_query": {
                    "endpoint": "POST /learning/precedents/query",
                    "description": "Search for relevant precedents"
                }
            }
        }
        result = embedding_service._extract_text(content)

        assert "TITLE: Phase 3 Features" in result
        assert "precedent_storage" in result
        assert "precedent_query" in result
        assert "POST /learning/precedents/store" in result
        assert "Store learnings" in result

    def test_extract_list(self, embedding_service):
        """Test extraction from list content."""
        content = {
            "title": "Tips",
            "items": ["First tip", "Second tip", "Third tip"]
        }
        result = embedding_service._extract_text(content)

        assert "TITLE: Tips" in result
        assert "First tip" in result
        assert "Second tip" in result
        assert "Third tip" in result

    def test_extract_deeply_nested(self, embedding_service):
        """Test extraction from deeply nested structure."""
        content = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "deep_value": "I am deeply nested"
                        }
                    }
                }
            }
        }
        result = embedding_service._extract_text(content)

        assert "deep_value" in result
        assert "I am deeply nested" in result

    def test_max_depth_protection(self, embedding_service):
        """Test that max depth prevents infinite recursion."""
        # Create deeply nested structure beyond max depth
        content = {"level": None}
        current = content
        for i in range(10):
            current["nested"] = {"level": i}
            current = current["nested"]

        # Should not raise, should truncate at max depth
        result = embedding_service._extract_text(content)
        assert isinstance(result, str)

    def test_extract_mixed_types(self, embedding_service):
        """Test extraction with mixed types (str, int, bool, list, dict)."""
        content = {
            "title": "Mixed Content",
            "count": 42,
            "enabled": True,
            "tags": ["api", "learning", "memory"],
            "config": {
                "timeout": 30,
                "retry": True
            }
        }
        result = embedding_service._extract_text(content)

        assert "TITLE: Mixed Content" in result
        assert "count: 42" in result
        assert "enabled: True" in result
        assert "api" in result
        assert "learning" in result
        assert "timeout: 30" in result

    def test_skip_private_fields(self, embedding_service):
        """Test that private fields (starting with _) are skipped."""
        content = {
            "title": "Public Title",
            "_internal": "Should be skipped",
            "_private_data": {"secret": "value"}
        }
        result = embedding_service._extract_text(content)

        assert "Public Title" in result
        assert "_internal" not in result
        assert "secret" not in result

    def test_skip_none_values(self, embedding_service):
        """Test that None values are skipped."""
        content = {
            "title": "Has Nulls",
            "empty": None,
            "also_null": None,
            "valid": "This exists"
        }
        result = embedding_service._extract_text(content)

        assert "Has Nulls" in result
        assert "This exists" in result
        assert "empty" not in result
        assert "also_null" not in result

    def test_header_fields_formatted(self, embedding_service):
        """Test that header fields are properly formatted with uppercase."""
        content = {
            "title": "My Title",
            "type": "decision",
            "severity": "high",
            "status": "completed"
        }
        result = embedding_service._extract_text(content)

        assert "TITLE: My Title" in result
        assert "TYPE: decision" in result
        assert "SEVERITY: high" in result
        assert "STATUS: completed" in result

    def test_priority_fields_extracted(self, embedding_service):
        """Test that priority content fields are extracted."""
        content = {
            "title": "Test",
            "summary": "A brief summary",
            "description": "Detailed description",
            "learning": "What we learned"
        }
        result = embedding_service._extract_text(content)

        assert "A brief summary" in result
        assert "Detailed description" in result
        assert "What we learned" in result

    def test_real_world_phase3_content(self, embedding_service):
        """Test with actual Phase 3 features content structure."""
        content = {
            "title": "Veris Audit Infrastructure - Phase 3 Features Released",
            "status": "production",
            "summary": "Agent learning infrastructure now available",
            "features": {
                "precedent_storage": {
                    "endpoint": "POST /learning/precedents/store",
                    "description": "Store learnings from completed tasks",
                    "precedent_types": ["failure", "success", "decision", "skill", "pattern"]
                },
                "precedent_query": {
                    "endpoint": "POST /learning/precedents/query",
                    "description": "Semantic search for relevant precedents"
                },
                "retention_processing": {
                    "endpoint": "POST /audit/retention/process",
                    "policies": {
                        "EPHEMERAL": "7 days then delete",
                        "TRACE": "30-90 days compress then delete",
                        "SCAR": "Forever - never delete"
                    }
                }
            },
            "usage_guide": {
                "before_task": "Query precedents for similar situations",
                "after_task": "Store learnings from outcomes"
            }
        }
        result = embedding_service._extract_text(content)

        # Verify key terms are extracted for searchability
        assert "Phase 3" in result
        assert "precedent" in result.lower()
        assert "learning" in result.lower()
        assert "retention" in result.lower()
        assert "EPHEMERAL" in result
        assert "SCAR" in result
        assert "semantic search" in result.lower()
        assert "query" in result.lower()


class TestExtractTextEdgeCases:
    """Edge case tests for _extract_text."""

    def test_empty_dict(self, embedding_service):
        """Test extraction from empty dict."""
        result = embedding_service._extract_text({})
        assert result == ""

    def test_empty_string(self, embedding_service):
        """Test extraction from empty string."""
        result = embedding_service._extract_text("")
        assert result == ""

    def test_none_value(self, embedding_service):
        """Test extraction from None."""
        result = embedding_service._extract_text(None)
        assert result == ""

    def test_integer_value(self, embedding_service):
        """Test extraction from integer."""
        result = embedding_service._extract_text(42)
        assert result == "42"

    def test_boolean_value(self, embedding_service):
        """Test extraction from boolean."""
        result = embedding_service._extract_text(True)
        assert result == "True"

    def test_empty_list(self, embedding_service):
        """Test extraction from empty list."""
        result = embedding_service._extract_text([])
        assert result == ""

    def test_list_of_dicts(self, embedding_service):
        """Test extraction from list of dicts."""
        content = [
            {"name": "Alice", "role": "developer"},
            {"name": "Bob", "role": "designer"}
        ]
        result = embedding_service._extract_text(content)

        assert "Alice" in result
        assert "Bob" in result
        assert "developer" in result
        assert "designer" in result
