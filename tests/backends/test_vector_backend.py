#!/usr/bin/env python3
"""
Comprehensive test suite for vector backend text extraction methods.

Tests all 5 extraction strategies and edge cases.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.backends.vector_backend import VectorBackend


class TestVectorBackendTextExtraction:
    """Test suite for _extract_text_content method covering all strategies."""

    @pytest.fixture
    def vector_backend(self):
        """Create a vector backend instance for testing."""
        mock_client = Mock()
        mock_embedding_generator = Mock()
        backend = VectorBackend(mock_client, mock_embedding_generator)
        return backend

    def test_strategy1_direct_text_field(self, vector_backend):
        """Test Strategy 1: Direct text fields extraction."""
        # Test with simple text field
        payload = {"text": "This is direct text content"}
        result = vector_backend._extract_text_content(payload)
        assert result == "This is direct text content"

        # Test with text field containing whitespace
        payload = {"text": "  Text with whitespace  "}
        result = vector_backend._extract_text_content(payload)
        assert result == "Text with whitespace"

        # Test with empty text field (should fall through)
        payload = {"text": ""}
        result = vector_backend._extract_text_content(payload)
        assert result == ""

    def test_strategy2_content_field_string(self, vector_backend):
        """Test Strategy 2: Content field as string."""
        payload = {"content": "Content as string"}
        result = vector_backend._extract_text_content(payload)
        assert result == "Content as string"

        # Test with content field with whitespace
        payload = {"content": "  Content with spaces  "}
        result = vector_backend._extract_text_content(payload)
        assert result == "Content with spaces"

    def test_strategy2_content_field_dict(self, vector_backend):
        """Test Strategy 2: Content field as nested dictionary."""
        # Test with nested text field
        payload = {"content": {"text": "Nested text content"}}
        result = vector_backend._extract_text_content(payload)
        assert result == "Nested text content"

        # Test with nested title field
        payload = {"content": {"title": "Nested title content"}}
        result = vector_backend._extract_text_content(payload)
        assert result == "Nested title content"

        # Test with nested description field
        payload = {"content": {"description": "Nested description"}}
        result = vector_backend._extract_text_content(payload)
        assert result == "Nested description"

        # Test priority: text > title > description
        payload = {"content": {"text": "Text", "title": "Title", "description": "Desc"}}
        result = vector_backend._extract_text_content(payload)
        assert result == "Text"

        payload = {"content": {"title": "Title", "description": "Desc"}}
        result = vector_backend._extract_text_content(payload)
        assert result == "Title"

    def test_strategy3_legacy_user_message(self, vector_backend):
        """Test Strategy 3: Legacy user_message field."""
        payload = {"user_message": "Legacy message content"}
        result = vector_backend._extract_text_content(payload)
        assert result == "Legacy message content"

        # Test with non-string user_message (should convert)
        payload = {"user_message": 12345}
        result = vector_backend._extract_text_content(payload)
        assert result == "12345"

        # Test with None user_message
        payload = {"user_message": None}
        result = vector_backend._extract_text_content(payload)
        assert result == "None"

    def test_strategy4_title_or_description(self, vector_backend):
        """Test Strategy 4: Title or description fields."""
        # Test title field
        payload = {"title": "Document Title"}
        result = vector_backend._extract_text_content(payload)
        assert result == "Document Title"

        # Test description field
        payload = {"description": "Document Description"}
        result = vector_backend._extract_text_content(payload)
        assert result == "Document Description"

        # Test priority: title > description
        payload = {"title": "Title", "description": "Description"}
        result = vector_backend._extract_text_content(payload)
        assert result == "Title"

        # Test with non-string values (should convert)
        payload = {"title": 999}
        result = vector_backend._extract_text_content(payload)
        assert result == "999"

    def test_strategy5_first_string_value(self, vector_backend):
        """Test Strategy 5: First string value in payload."""
        # Test finding first valid string
        payload = {
            "id": "123",  # Should be skipped
            "type": "doc",  # Should be skipped
            "namespace": "test",  # Should be skipped
            "random_field": "This should be extracted",
        }
        result = vector_backend._extract_text_content(payload)
        assert result == "This should be extracted"

        # Test with multiple string fields (should get first valid)
        payload = {
            "field1": "",  # Empty, should skip
            "field2": "First valid content",
            "field3": "Second content",
        }
        result = vector_backend._extract_text_content(payload)
        assert result == "First valid content"

    def test_edge_cases_empty_payload(self, vector_backend):
        """Test edge cases with empty or malformed payloads."""
        # Empty payload
        payload = {}
        result = vector_backend._extract_text_content(payload)
        assert result == ""

        # Payload with only excluded fields
        payload = {"id": "123", "type": "doc", "namespace": "test"}
        result = vector_backend._extract_text_content(payload)
        assert result == ""

        # Payload with None values
        payload = {"text": None, "content": None, "title": None}
        result = vector_backend._extract_text_content(payload)
        assert result == ""

    def test_edge_cases_complex_nested(self, vector_backend):
        """Test complex nested structures."""
        # Deeply nested content
        payload = {
            "content": {"data": {"text": "Should not reach this"}, "title": "Should extract this"}
        }
        result = vector_backend._extract_text_content(payload)
        assert result == "Should extract this"

        # Content with empty nested dict
        payload = {"content": {}}
        result = vector_backend._extract_text_content(payload)
        assert result == ""

    def test_edge_cases_type_handling(self, vector_backend):
        """Test various data type handling."""
        # Boolean values
        payload = {"text": True}
        result = vector_backend._extract_text_content(payload)
        assert result == ""  # Not a string, should skip

        # List values
        payload = {"content": ["item1", "item2"]}
        result = vector_backend._extract_text_content(payload)
        assert result == ""  # Not string or dict, should skip

        # Integer as content
        payload = {"content": 42}
        result = vector_backend._extract_text_content(payload)
        assert result == ""  # Not string or dict, should skip

    def test_extraction_priority_order(self, vector_backend):
        """Test the complete priority order of extraction strategies."""
        # All strategies present - should use text field (Strategy 1)
        payload = {
            "text": "Direct text",
            "content": "Content string",
            "user_message": "Legacy message",
            "title": "Title",
            "description": "Description",
            "other_field": "Other content",
        }
        result = vector_backend._extract_text_content(payload)
        assert result == "Direct text"

        # No text field - should use content (Strategy 2)
        payload = {
            "content": "Content string",
            "user_message": "Legacy message",
            "title": "Title",
            "other_field": "Other content",
        }
        result = vector_backend._extract_text_content(payload)
        assert result == "Content string"

        # No text or content - should use user_message (Strategy 3)
        payload = {
            "user_message": "Legacy message",
            "title": "Title",
            "other_field": "Other content",
        }
        result = vector_backend._extract_text_content(payload)
        assert result == "Legacy message"

        # No text, content, or user_message - should use title (Strategy 4)
        payload = {"title": "Title", "description": "Description", "other_field": "Other content"}
        result = vector_backend._extract_text_content(payload)
        assert result == "Title"

        # Only other fields - should use first valid (Strategy 5)
        payload = {
            "id": "skip",
            "type": "skip",
            "other_field": "Other content",
            "another_field": "Another content",
        }
        result = vector_backend._extract_text_content(payload)
        assert result == "Other content"

    def test_whitespace_handling(self, vector_backend):
        """Test proper whitespace stripping across all strategies."""
        test_cases = [
            {"text": "  spaces around  "},
            {"content": "  spaces around  "},
            {"user_message": "  spaces around  "},
            {"title": "  spaces around  "},
            {"description": "  spaces around  "},
            {"random": "  spaces around  "},
        ]

        for payload in test_cases:
            result = vector_backend._extract_text_content(payload)
            assert result == "spaces around", f"Failed for payload: {payload}"

    def test_unicode_and_special_chars(self, vector_backend):
        """Test handling of unicode and special characters."""
        # Unicode text
        payload = {"text": "Hello 世界 🌍 émojis"}
        result = vector_backend._extract_text_content(payload)
        assert result == "Hello 世界 🌍 émojis"

        # Special characters
        payload = {"content": "Text with\nnewlines\tand\ttabs"}
        result = vector_backend._extract_text_content(payload)
        assert result == "Text with\nnewlines\tand\ttabs"

        # HTML/XML content (should not be parsed)
        payload = {"text": "<p>HTML content</p>"}
        result = vector_backend._extract_text_content(payload)
        assert result == "<p>HTML content</p>"


class TestVectorBackendSearchByEmbedding:
    """Test suite for search_by_embedding method (HyDE support)."""

    @pytest.fixture
    def vector_backend(self):
        """Create a vector backend instance for testing."""
        mock_client = Mock()
        mock_embedding_generator = Mock()
        backend = VectorBackend(mock_client, mock_embedding_generator)
        return backend

    @pytest.fixture
    def mock_qdrant_results(self):
        """Create mock Qdrant search results."""
        result1 = Mock()
        result1.id = "doc_1"
        result1.score = 0.95
        result1.payload = {"text": "Document about Neo4j configuration", "type": "documentation"}

        result2 = Mock()
        result2.id = "doc_2"
        result2.score = 0.82
        result2.payload = {"text": "Redis setup guide", "type": "documentation"}

        return [result1, result2]

    @pytest.mark.asyncio
    async def test_search_by_embedding_success(self, vector_backend, mock_qdrant_results):
        """Test successful search by embedding."""
        from src.interfaces.backend_interface import SearchOptions

        # Mock the client search method
        vector_backend.client.search = Mock(return_value=mock_qdrant_results)

        embedding = [0.1, 0.2, 0.3] * 512  # Simulated 1536-dim embedding
        options = SearchOptions(limit=5)

        results = await vector_backend.search_by_embedding(embedding, options)

        assert len(results) == 2
        assert results[0].id == "doc_1"
        assert results[0].score == 0.95
        assert "Neo4j" in results[0].text
        # Verify client.search was called with the embedding
        vector_backend.client.search.assert_called_once()
        call_args = vector_backend.client.search.call_args
        assert call_args.kwargs["query_vector"] == embedding

    @pytest.mark.asyncio
    async def test_search_by_embedding_empty_results(self, vector_backend):
        """Test search by embedding with no results."""
        from src.interfaces.backend_interface import SearchOptions

        vector_backend.client.search = Mock(return_value=[])

        embedding = [0.1, 0.2, 0.3]
        options = SearchOptions(limit=10)

        results = await vector_backend.search_by_embedding(embedding, options)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_by_embedding_with_score_threshold(self, vector_backend, mock_qdrant_results):
        """Test search by embedding respects score threshold."""
        from src.interfaces.backend_interface import SearchOptions

        vector_backend.client.search = Mock(return_value=mock_qdrant_results)

        embedding = [0.1, 0.2, 0.3]
        options = SearchOptions(limit=10, score_threshold=0.9)

        results = await vector_backend.search_by_embedding(embedding, options)

        # Should only include results above 0.9 threshold
        assert all(r.score >= 0.9 for r in results)

    @pytest.mark.asyncio
    async def test_search_by_embedding_error_handling(self, vector_backend):
        """Test error handling in search by embedding."""
        from src.interfaces.backend_interface import SearchOptions, BackendSearchError

        vector_backend.client.search = Mock(side_effect=Exception("Connection failed"))

        embedding = [0.1, 0.2, 0.3]
        options = SearchOptions(limit=5)

        with pytest.raises(BackendSearchError) as exc_info:
            await vector_backend.search_by_embedding(embedding, options)

        assert "vector" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_by_embedding_logs_metadata(self, vector_backend, mock_qdrant_results):
        """Test that search by embedding logs appropriate metadata."""
        from src.interfaces.backend_interface import SearchOptions

        vector_backend.client.search = Mock(return_value=mock_qdrant_results)

        embedding = [0.1] * 1536
        options = SearchOptions(limit=5)

        results = await vector_backend.search_by_embedding(embedding, options)

        # Verify metadata indicates this was a HyDE search
        assert len(results) == 2
        # search_by_embedding should mark results appropriately
        for result in results:
            assert result.metadata.get("vector_search") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
