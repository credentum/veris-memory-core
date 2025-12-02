#!/usr/bin/env python3
"""
Test metadata field separation in retrieve_context responses.

This module contains tests that validate the proper separation of metadata fields
from content fields in the retrieve_context endpoint responses. This fix is critical
for the S2 golden fact recall checks which rely on the metadata.golden_fact field
to identify golden facts for testing recall capabilities.

The tests ensure that:
- Metadata fields are returned in a separate 'metadata' key
- Content fields remain in the 'content' key
- S2 checks can successfully identify golden facts
- Both vector and graph search modes properly separate metadata
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock the required modules and classes
@pytest.fixture
def mock_request():
    """Create a mock retrieve context request."""
    request = Mock()
    request.query = "test query"
    request.limit = 10
    request.search_mode = "hybrid"
    request.sort_by = Mock(value="relevance")
    request.type = "all"
    return request

@pytest.fixture
def mock_neo4j_results_with_metadata():
    """Create mock Neo4j results that include metadata fields."""
    return [
        {
            'n': {
                'id': 'test-123',
                'title': 'Test Context',
                'description': 'A test context with metadata',
                'type': 'decision',
                'author': 'test_user',
                'created_at': '2025-01-01T00:00:00',
                # Metadata fields that should be separated
                'golden_fact': True,
                'category': 'test',
                'priority': 'high',
                'sprint': '13'
            }
        }
    ]

def test_retrieve_context_includes_metadata_field():
    """
    Test that retrieve_context response includes separate metadata field.

    This test validates that the endpoint properly separates metadata
    fields from content fields in the response. It ensures that:
    - Results include both 'content' and 'metadata' keys
    - Golden fact flag is placed in metadata, not content
    - Other metadata fields (category, priority) are properly separated

    This fix is essential for S2 golden fact recall checks to work correctly.
    """

    from src.mcp_server.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # Mock the dependencies
    with patch('src.mcp_server.main.neo4j_client') as mock_neo4j:
        with patch('src.mcp_server.main.qdrant_client', None):  # Disable vector search
            with patch('src.mcp_server.main.retrieval_core', None):  # Disable unified search

                # Setup mock to return our test data
                mock_neo4j.query.return_value = [
                    {
                        'n': {
                            'id': 'test-123',
                            'title': 'Golden Fact Test',
                            'type': 'decision',
                            'golden_fact': True,
                            'category': 'architecture',
                            'priority': 'high'
                        }
                    }
                ]

                # Make request
                response = client.post(
                    "/tools/retrieve_context",
                    json={
                        "query": "test",
                        "limit": 10
                    }
                )

                assert response.status_code == 200
                data = response.json()

                # Verify response structure
                assert "results" in data
                assert len(data["results"]) > 0

                result = data["results"][0]

                # Key assertions: metadata should be separate
                assert "metadata" in result, "Result should have metadata field"
                assert "content" in result, "Result should have content field"

                # Metadata fields should be in metadata, not content
                metadata = result["metadata"]
                content = result["content"]

                assert metadata.get("golden_fact") == True, "golden_fact should be in metadata"
                assert metadata.get("category") == "architecture", "category should be in metadata"
                assert metadata.get("priority") == "high", "priority should be in metadata"

                # Content fields should not include metadata fields
                assert "golden_fact" not in content, "golden_fact should not be in content"
                assert "category" not in content, "category should not be in content"
                assert "priority" not in content, "priority should not be in content"

def test_s2_golden_fact_recall_can_find_metadata():
    """
    Test that S2 checks can find golden facts using metadata field.

    This test simulates the exact behavior of the S2 golden fact recall check
    by filtering results based on metadata.golden_fact == True. It validates:
    - Golden facts can be identified using the metadata field
    - Non-golden facts are properly excluded
    - The filtering logic matches S2 implementation requirements

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If golden facts cannot be properly identified
    """
    mock_results = [
        {
            "id": "fact-1",
            "content": {"title": "Important Fact", "description": "Test"},
            "metadata": {"golden_fact": True, "category": "test"},
            "score": 0.95,
            "source": "graph"
        },
        {
            "id": "fact-2",
            "content": {"title": "Regular Context", "description": "Not golden"},
            "metadata": {"category": "normal"},
            "score": 0.80,
            "source": "graph"
        },
        {
            "id": "fact-3",
            "content": {"title": "Another Golden Fact", "description": "Important"},
            "metadata": {"golden_fact": True, "priority": "high"},
            "score": 0.90,
            "source": "graph"
        }
    ]

    # Filter for golden facts (what S2 check does)
    golden_facts = [
        r for r in mock_results
        if r.get("metadata", {}).get("golden_fact") == True
    ]

    assert len(golden_facts) == 2, "Should find 2 golden facts"
    assert golden_facts[0]["id"] == "fact-1"
    assert golden_facts[1]["id"] == "fact-3"

def test_vector_search_includes_metadata():
    """
    Test that vector search results also include metadata field.

    This test ensures that vector search (Qdrant) results properly separate
    metadata fields from content, maintaining consistency across all search modes.
    It validates:
    - Vector search results include 'metadata' key
    - Metadata fields from Qdrant payload are properly extracted
    - Consistency with graph search result format

    Vector search is part of hybrid search mode, so this fix ensures
    metadata separation works regardless of which backend provides results.
    """

    from src.mcp_server.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    with patch('src.mcp_server.main.qdrant_client') as mock_qdrant:
        with patch('src.mcp_server.main.neo4j_client', None):
            with patch('src.mcp_server.main.retrieval_core', None):
                with patch('src.mcp_server.main.generate_embedding') as mock_embed:

                    # Mock embedding generation
                    mock_embed.return_value = [0.1] * 768

                    # Mock Qdrant search results
                    mock_result = Mock()
                    mock_result.id = "vector-123"
                    mock_result.score = 0.95
                    mock_result.payload = {
                        "content": {"title": "Vector Result"},
                        "metadata": {"golden_fact": True, "source": "vector_db"},
                        "type": "decision"
                    }

                    mock_qdrant.search.return_value = [mock_result]

                    response = client.post(
                        "/tools/retrieve_context",
                        json={
                            "query": "test",
                            "limit": 10,
                            "search_mode": "vector"
                        }
                    )

                    assert response.status_code == 200
                    data = response.json()

                    result = data["results"][0]
                    assert "metadata" in result
                    assert result["metadata"]["golden_fact"] == True
                    assert result["metadata"]["source"] == "vector_db"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
