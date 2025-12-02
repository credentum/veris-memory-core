#!/usr/bin/env python3
"""
Comprehensive test suite for graph backend search functionality.

Tests the _build_search_query method including:
- Text search across all actual Context fields
- NULL safety checks for missing fields
- Case-insensitive search with toLower()
- Search with namespace filters
- Search with type/user_id filters
- Index creation for actual fields
"""

from unittest.mock import Mock, MagicMock, call
import pytest

from src.backends.graph_backend import GraphBackend
from src.interfaces.backend_interface import SearchOptions


class TestGraphBackendSearchQuery:
    """Test suite for _build_search_query method."""

    @pytest.fixture
    def graph_backend(self):
        """Create a graph backend instance for testing."""
        mock_client = Mock()
        mock_client.query = Mock(return_value=[])
        backend = GraphBackend(mock_client)
        return backend

    def test_build_search_query_basic(self, graph_backend):
        """Test basic search query generation with text search."""
        query = "test"  # Single word to test basic search
        options = SearchOptions(limit=10)

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Verify parameters
        assert parameters["search_text"] == "test"
        assert parameters["limit"] == 10

        # Verify query structure
        assert "MATCH" in cypher_query
        assert "Context" in cypher_query
        assert "WHERE" in cypher_query
        assert "RETURN n" in cypher_query
        assert "ORDER BY n.timestamp DESC" in cypher_query
        assert "LIMIT $limit" in cypher_query

    def test_search_all_actual_fields(self, graph_backend):
        """Test that search query includes all actual Context node fields."""
        query = "banana"
        options = SearchOptions(limit=5)

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Verify all actual fields are searched
        assert "n.title" in cypher_query
        assert "n.description" in cypher_query
        assert "n.keyword" in cypher_query
        assert "n.user_input" in cypher_query
        assert "n.bot_response" in cypher_query
        assert "n.searchable_text" in cypher_query  # PR #340

        # Verify old non-existent fields are NOT in query
        assert "n.text" not in cypher_query
        assert "n.content" not in cypher_query or "CONTAINS" not in cypher_query.split("n.content")[0]

    def test_null_safety_checks(self, graph_backend):
        """Test that NULL safety checks are present for all fields."""
        query = "test"
        options = SearchOptions(limit=5)

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Verify NULL checks for each field
        assert "n.title IS NOT NULL" in cypher_query
        assert "n.description IS NOT NULL" in cypher_query
        assert "n.keyword IS NOT NULL" in cypher_query
        assert "n.user_input IS NOT NULL" in cypher_query
        assert "n.bot_response IS NOT NULL" in cypher_query
        assert "n.searchable_text IS NOT NULL" in cypher_query  # PR #340

    def test_case_insensitive_search(self, graph_backend):
        """Test that search uses toLower() for case-insensitive matching."""
        query = "TeSt"  # Single word
        options = SearchOptions(limit=5)

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Verify toLower() is used for case-insensitive search
        assert "toLower(n.title)" in cypher_query
        assert "toLower(n.description)" in cypher_query
        assert "toLower($search_text)" in cypher_query

        # Verify CONTAINS is used (not exact match)
        assert "CONTAINS" in cypher_query

    def test_search_with_namespace_filter(self, graph_backend):
        """Test search query with namespace filter."""
        query = "test"
        options = SearchOptions(limit=5, namespace="/project/test")

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Verify namespace filter
        assert "n.namespace = $namespace" in cypher_query
        assert parameters["namespace"] == "/project/test"

        # Verify AND logic combines filters
        assert "AND" in cypher_query

    def test_search_with_type_filter(self, graph_backend):
        """Test search query with type filter."""
        query = "test"
        options = SearchOptions(
            limit=5,
            filters={"type": "decision"}
        )

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Verify type filter
        assert "n.type = $type_filter" in cypher_query
        assert parameters["type_filter"] == "decision"

    def test_search_with_user_id_filter(self, graph_backend):
        """Test search query with user_id filter."""
        query = "test"
        options = SearchOptions(
            limit=5,
            filters={"user_id": "user123"}
        )

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Verify user_id filter
        assert "n.user_id = $user_id_filter" in cypher_query
        assert parameters["user_id_filter"] == "user123"

    def test_search_with_multiple_filters(self, graph_backend):
        """Test search query with multiple filters combined."""
        query = "test"
        options = SearchOptions(
            limit=10,
            namespace="/project/test",
            filters={
                "type": "trace",
                "user_id": "voice_bot"
            }
        )

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Verify all filters present
        assert "n.namespace = $namespace" in cypher_query
        assert "n.type = $type_filter" in cypher_query
        assert "n.user_id = $user_id_filter" in cypher_query

        # Verify all parameters
        assert parameters["namespace"] == "/project/test"
        assert parameters["type_filter"] == "trace"
        assert parameters["user_id_filter"] == "voice_bot"
        assert parameters["limit"] == 10

    def test_search_query_or_logic(self, graph_backend):
        """Test that text fields are combined with OR logic."""
        query = "banana"
        options = SearchOptions(limit=5)

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Verify OR logic between text fields
        assert cypher_query.count(" OR ") >= 4  # At least 4 ORs for 5 fields


class TestGraphBackendIndexes:
    """Test suite for index creation."""

    @pytest.fixture
    def graph_backend(self):
        """Create a graph backend instance for testing."""
        mock_client = Mock()
        mock_client.query = Mock(return_value=[])
        backend = GraphBackend(mock_client)
        return backend

    def test_create_indexes_for_actual_fields(self, graph_backend):
        """Test that indexes are created for actual Context fields."""
        graph_backend._create_indexes()

        # Get all index creation queries called
        calls = graph_backend.client.query.call_args_list
        index_queries = [call[0][0] for call in calls]

        # Verify indexes for searchable fields
        assert any("n.title" in query for query in index_queries)
        assert any("n.description" in query for query in index_queries)
        assert any("n.keyword" in query for query in index_queries)
        assert any("n.user_input" in query for query in index_queries)
        assert any("n.bot_response" in query for query in index_queries)
        assert any("n.searchable_text" in query for query in index_queries)  # PR #340

        # Verify indexes for filter fields
        assert any("n.timestamp" in query for query in index_queries)
        assert any("n.namespace" in query for query in index_queries)
        assert any("n.type" in query for query in index_queries)

    def test_no_indexes_for_nonexistent_fields(self, graph_backend):
        """Test that indexes are NOT created for non-existent text/content fields."""
        graph_backend._create_indexes()

        # Get all index creation queries called
        calls = graph_backend.client.query.call_args_list
        index_queries = [call[0][0] for call in calls]

        # Verify NO indexes for non-existent fields
        # (text and content may appear in Context but not as indexed fields)
        text_content_indexes = [
            query for query in index_queries
            if ("ON (n.text)" in query or "ON (n.content)" in query)
        ]
        assert len(text_content_indexes) == 0, "Should not create indexes for non-existent text/content fields"


class TestGraphBackendTextExtraction:
    """Test suite for _convert_to_memory_results text extraction."""

    @pytest.fixture
    def graph_backend(self):
        """Create a graph backend instance for testing."""
        mock_client = Mock()
        backend = GraphBackend(mock_client)
        return backend

    def test_extract_text_from_title(self, graph_backend):
        """Test text extraction prioritizes title field."""
        raw_results = [{
            'n': {
                'id': 'test-id-1',
                'title': 'Test Title',
                'type': 'decision'
            }
        }]

        results = graph_backend._convert_to_memory_results(raw_results)

        assert len(results) == 1
        assert results[0].text == 'Test Title'

    def test_extract_text_from_description(self, graph_backend):
        """Test text extraction falls back to description."""
        raw_results = [{
            'n': {
                'id': 'test-id-2',
                'description': 'Test Description',
                'type': 'decision'
            }
        }]

        results = graph_backend._convert_to_memory_results(raw_results)

        assert len(results) == 1
        assert results[0].text == 'Test Description'

    def test_extract_text_from_user_input(self, graph_backend):
        """Test text extraction works with voice bot user_input field."""
        raw_results = [{
            'n': {
                'id': 'test-id-3',
                'user_input': 'You are my sunshine',
                'type': 'trace'
            }
        }]

        results = graph_backend._convert_to_memory_results(raw_results)

        assert len(results) == 1
        assert results[0].text == 'You are my sunshine'

    def test_extract_text_from_bot_response(self, graph_backend):
        """Test text extraction works with voice bot bot_response field."""
        raw_results = [{
            'n': {
                'id': 'test-id-4',
                'bot_response': 'You make me happy',
                'type': 'trace'
            }
        }]

        results = graph_backend._convert_to_memory_results(raw_results)

        assert len(results) == 1
        assert results[0].text == 'You make me happy'

    def test_extract_text_from_keyword(self, graph_backend):
        """Test text extraction works with keyword field."""
        raw_results = [{
            'n': {
                'id': 'test-id-5',
                'keyword': 'banana',
                'type': 'log'
            }
        }]

        results = graph_backend._convert_to_memory_results(raw_results)

        assert len(results) == 1
        assert results[0].text == 'banana'

    def test_text_extraction_priority_order(self, graph_backend):
        """Test that text extraction respects priority order."""
        # Title should have highest priority
        raw_results = [{
            'n': {
                'id': 'test-id-6',
                'title': 'High Priority Title',
                'description': 'Lower Priority Description',
                'keyword': 'Lowest Priority Keyword',
                'type': 'decision'
            }
        }]

        results = graph_backend._convert_to_memory_results(raw_results)

        assert len(results) == 1
        assert results[0].text == 'High Priority Title'

    def test_no_extraction_from_nonexistent_fields(self, graph_backend):
        """Test that extraction doesn't rely on non-existent text/content fields."""
        # Node with only non-existent fields should be skipped
        raw_results = [{
            'n': {
                'id': 'test-id-7',
                'type': 'decision'
                # No text-extractable fields
            }
        }]

        results = graph_backend._convert_to_memory_results(raw_results)

        # Should skip node with no text content
        assert len(results) == 0

    def test_legacy_field_support(self, graph_backend):
        """Test that legacy user_message and message fields still work."""
        raw_results = [{
            'n': {
                'id': 'test-id-8',
                'user_message': 'Legacy user message',
                'type': 'log'
            }
        }]

        results = graph_backend._convert_to_memory_results(raw_results)

        assert len(results) == 1
        assert results[0].text == 'Legacy user message'


class TestGraphBackendEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def graph_backend(self):
        """Create a graph backend instance for testing."""
        mock_client = Mock()
        backend = GraphBackend(mock_client)
        return backend

    def test_empty_query_string(self, graph_backend):
        """Test behavior with empty query string."""
        query = ""
        options = SearchOptions(limit=5)

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Should still generate valid query
        assert "MATCH" in cypher_query
        # Empty query should result in "false" condition (no search_text parameter)
        assert "false" in cypher_query

    def test_query_with_special_characters(self, graph_backend):
        """Test query with special characters."""
        query = "test\"query'with$special#chars"
        options = SearchOptions(limit=5)

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        # Should pass special characters as parameter (safe from injection)
        assert parameters["search_text"] == query

    def test_large_limit_value(self, graph_backend):
        """Test with large limit value (within SearchOptions max of 1000)."""
        query = "test"
        options = SearchOptions(limit=1000)  # Max allowed by SearchOptions

        cypher_query, parameters = graph_backend._build_search_query(query, options)

        assert parameters["limit"] == 1000

    def test_node_with_null_fields(self, graph_backend):
        """Test handling of nodes with explicitly NULL fields."""
        raw_results = [{
            'n': {
                'id': 'test-id-9',
                'title': None,
                'description': None,
                'keyword': None,
                'user_input': 'Valid input',
                'bot_response': None,
                'type': 'trace'
            }
        }]

        results = graph_backend._convert_to_memory_results(raw_results)

        # Should extract from user_input (first non-null field)
        assert len(results) == 1
        assert results[0].text == 'Valid input'
