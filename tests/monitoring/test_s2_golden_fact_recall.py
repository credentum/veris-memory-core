#!/usr/bin/env python3
"""
Unit tests for S2 Golden Fact Recall check.

Tests the fixes for issue #279:
- Author field stored in metadata (not top-level)
- metadata_filters parameter used for retrieval (not filters)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.monitoring.sentinel.checks.s2_golden_fact_recall import GoldenFactRecall
from src.monitoring.sentinel.models import SentinelConfig


@pytest.fixture
def mock_config():
    """Create mock sentinel config."""
    config = Mock(spec=SentinelConfig)
    config.target_base_url = "http://localhost:8000"
    config.api_key = "test_key"
    config.get = Mock(return_value=None)
    return config


@pytest.fixture
def golden_fact_check(mock_config):
    """Create GoldenFactRecall check instance."""
    return GoldenFactRecall(mock_config)


@pytest.mark.asyncio
async def test_store_fact_puts_author_in_metadata(golden_fact_check):
    """Test that _store_fact puts author in metadata, not top-level (issue #279 fix)."""
    mock_session = AsyncMock()

    # Mock the API call to capture the payload
    captured_payload = None

    async def capture_test_api_call(session, method, url, data, expected_status, timeout):
        nonlocal captured_payload
        captured_payload = data
        return True, "Success", 10.0, {"success": True, "id": "ctx_test123"}

    with patch.object(golden_fact_check, 'test_api_call', side_effect=capture_test_api_call):
        result = await golden_fact_check._store_fact(
            mock_session,
            {"name": "Matt"},
            "test_user_123"
        )

    # Verify author is in metadata, not at top level
    assert captured_payload is not None, "API call should have been made"
    assert "author" not in captured_payload, "Author should NOT be at top-level"
    assert "metadata" in captured_payload, "Metadata field should exist"
    assert captured_payload["metadata"]["author"] == "test_user_123", "Author should be in metadata"

    # Verify other metadata fields are preserved
    assert captured_payload["metadata"]["test_type"] == "golden_recall"
    assert captured_payload["metadata"]["sentinel"] is True
    assert captured_payload["metadata"]["content_type"] == "fact"

    # Verify content and type are correct
    assert captured_payload["content"] == {"name": "Matt"}
    assert captured_payload["type"] == "log"


@pytest.mark.asyncio
async def test_test_recall_uses_metadata_filters_parameter(golden_fact_check):
    """Test that _test_recall uses metadata_filters parameter, not filters (issue #279 fix)."""
    mock_session = AsyncMock()

    # Mock the API call to capture the payload
    captured_payload = None

    async def capture_test_api_call(session, method, url, data, expected_status, timeout):
        nonlocal captured_payload
        captured_payload = data
        return True, "Success", 15.0, {
            "success": True,
            "results": [
                {
                    "id": "ctx_test123",
                    "content": {"name": "Matt"},
                    "score": 0.95
                }
            ]
        }

    with patch.object(golden_fact_check, 'test_api_call', side_effect=capture_test_api_call):
        result = await golden_fact_check._test_recall(
            mock_session,
            "What's my name?",
            "Matt",
            "test_user_123"
        )

    # Verify metadata_filters is used, not filters
    assert captured_payload is not None, "API call should have been made"
    assert "filters" not in captured_payload, "Should NOT use 'filters' parameter"
    assert "metadata_filters" in captured_payload, "Should use 'metadata_filters' parameter"
    assert captured_payload["metadata_filters"]["author"] == "test_user_123", "Should filter by author"

    # Verify other query parameters
    assert captured_payload["query"] == "What's my name?"
    assert captured_payload["limit"] == 5


@pytest.mark.asyncio
async def test_user_isolation_with_metadata_filters(golden_fact_check):
    """Test that metadata_filters properly isolate users' test data."""
    mock_session = AsyncMock()

    # Simulate storing facts for two different users
    user1_id = "sentinel_test_user1"
    user2_id = "sentinel_test_user2"

    # Mock store calls
    async def mock_store_call(session, method, url, data, expected_status, timeout):
        return True, "Success", 10.0, {"success": True, "id": f"ctx_{data['metadata']['author']}"}

    # Mock retrieve calls - return data matching the filter
    async def mock_retrieve_call(session, method, url, data, expected_status, timeout):
        metadata_filters = data.get("metadata_filters", {})
        author_filter = metadata_filters.get("author")

        if author_filter == user1_id:
            return True, "Success", 15.0, {
                "success": True,
                "results": [
                    {"id": "ctx_user1", "content": {"name": "Alice"}, "score": 0.95}
                ]
            }
        elif author_filter == user2_id:
            return True, "Success", 15.0, {
                "success": True,
                "results": [
                    {"id": "ctx_user2", "content": {"name": "Bob"}, "score": 0.95}
                ]
            }
        else:
            # No filter or wrong filter - should not happen with our fix
            return True, "Success", 15.0, {"success": True, "results": []}

    with patch.object(golden_fact_check, 'test_api_call') as mock_api:
        # Route calls to appropriate mock based on URL
        async def side_effect_router(session, method, url, data, expected_status, timeout):
            if "store_context" in url:
                return await mock_store_call(session, method, url, data, expected_status, timeout)
            elif "retrieve_context" in url:
                return await mock_retrieve_call(session, method, url, data, expected_status, timeout)

        mock_api.side_effect = side_effect_router

        # Store and retrieve for user1
        store_result1 = await golden_fact_check._store_fact(mock_session, {"name": "Alice"}, user1_id)
        recall_result1 = await golden_fact_check._test_recall(mock_session, "What's my name?", "Alice", user1_id)

        # Store and retrieve for user2
        store_result2 = await golden_fact_check._store_fact(mock_session, {"name": "Bob"}, user2_id)
        recall_result2 = await golden_fact_check._test_recall(mock_session, "What's my name?", "Bob", user2_id)

    # Verify user1 gets only their data
    assert recall_result1["success"] is True
    assert "Alice" in str(recall_result1["response"])
    assert "Bob" not in str(recall_result1["response"])

    # Verify user2 gets only their data
    assert recall_result2["success"] is True
    assert "Bob" in str(recall_result2["response"])
    assert "Alice" not in str(recall_result2["response"])


@pytest.mark.asyncio
async def test_full_check_passes_with_proper_isolation(golden_fact_check):
    """Test that the full S2 check passes when data is properly isolated."""

    # Mock aiohttp session
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        # Mock test_api_call to simulate successful store, retrieve, and cleanup (PR #399)
        call_count = {"store": 0, "retrieve": 0, "cleanup": 0}

        async def mock_test_api_call(session, method, url, data, expected_status, timeout):
            if "store_context" in url:
                call_count["store"] += 1
                return True, "Success", 10.0, {"success": True, "id": f"ctx_test_{call_count['store']}"}

            elif "retrieve_context" in url:
                call_count["retrieve"] += 1
                # Return the expected content based on the query
                query = data.get("query", "")

                # Semantic search test cases
                if "name" in query.lower():
                    return True, "Success", 15.0, {
                        "success": True,
                        "results": [{"id": "ctx_1", "content": {"name": "Matt"}, "score": 0.95}]
                    }
                elif "food" in query.lower():
                    return True, "Success", 15.0, {
                        "success": True,
                        "results": [{"id": "ctx_2", "content": {"food": "spicy"}, "score": 0.95}]
                    }
                elif "live" in query.lower() or "location" in query.lower():
                    return True, "Success", 15.0, {
                        "success": True,
                        "results": [{"id": "ctx_3", "content": {"location": "San Francisco"}, "score": 0.95}]
                    }
                # Graph relationship test cases (PR #399: 3 additional test cases)
                elif "technology" in query.lower() or "veris memory" in query.lower() or "implemented" in query.lower():
                    return True, "Success", 15.0, {
                        "success": True,
                        "results": [{"id": "ctx_4", "content": {"project": "Veris Memory", "tech_stack": "Python FastAPI Neo4j"}, "score": 0.95}]
                    }
                elif "semantic search" in query.lower() or "depend" in query.lower() or "component" in query.lower():
                    return True, "Success", 15.0, {
                        "success": True,
                        "results": [{"id": "ctx_5", "content": {"feature": "semantic search", "dependencies": "Qdrant vector database embeddings"}, "score": 0.95}]
                    }
                elif "monitoring" in query.lower() or "related" in query.lower() or "connect" in query.lower():
                    return True, "Success", 15.0, {
                        "success": True,
                        "results": [{"id": "ctx_6", "content": {"component": "monitoring", "related_systems": "Sentinel checks metrics Neo4j"}, "score": 0.95}]
                    }
                else:
                    return True, "Success", 15.0, {"success": True, "results": []}

            elif "query_graph" in url:
                # PR #399: Handle cleanup calls
                call_count["cleanup"] += 1
                return True, "Success", 5.0, {"results": [{"deleted": 1}]}

            return False, "Unknown URL", 0, None

        with patch.object(golden_fact_check, 'test_api_call', side_effect=mock_test_api_call):
            result = await golden_fact_check.run_check()

    # Verify the check passes with good success rate
    assert result.status == "pass", f"Check should pass, got: {result.status}, message: {result.message}"
    assert result.details["success_rate"] >= 0.8, "Success rate should be >= 80%"
    assert result.details["passed_tests"] >= result.details["total_tests"] * 0.8, "Most tests should pass"
    # PR #399: Verify cleanup was called for each test case
    assert call_count["cleanup"] == 6, f"Cleanup should be called 6 times (once per test case), got {call_count['cleanup']}"


@pytest.mark.asyncio
async def test_payload_structure_matches_api_contract(golden_fact_check):
    """Test that payloads match the MCP API contracts (store_context.json and retrieve_context.json)."""
    mock_session = AsyncMock()

    # Test store_context payload
    store_payload = None
    async def capture_store(session, method, url, data, expected_status, timeout):
        nonlocal store_payload
        store_payload = data
        return True, "Success", 10.0, {"success": True, "id": "ctx_test"}

    with patch.object(golden_fact_check, 'test_api_call', side_effect=capture_store):
        await golden_fact_check._store_fact(mock_session, {"key": "value"}, "test_user")

    # Verify store_context contract compliance
    # Required fields: content, type
    assert "content" in store_payload
    assert "type" in store_payload
    # Optional fields: metadata, relationships
    assert "metadata" in store_payload
    # Fields NOT in contract (should not be present at top level):
    assert "author" not in store_payload, "author is not a valid top-level field per store_context.json"

    # Test retrieve_context payload
    retrieve_payload = None
    async def capture_retrieve(session, method, url, data, expected_status, timeout):
        nonlocal retrieve_payload
        retrieve_payload = data
        return True, "Success", 15.0, {"success": True, "results": []}

    with patch.object(golden_fact_check, 'test_api_call', side_effect=capture_retrieve):
        await golden_fact_check._test_recall(mock_session, "test query", "expected", "test_user")

    # Verify retrieve_context contract compliance
    # Required fields: query
    assert "query" in retrieve_payload
    # Optional fields per contract: type, search_mode, limit, filters, include_relationships, sort_by
    # GraphRAG additions: metadata_filters (not in base contract but implemented)
    assert "limit" in retrieve_payload
    # Verify we use metadata_filters for custom metadata (implementation detail)
    assert "metadata_filters" in retrieve_payload, "Should use metadata_filters for author filtering"
    # The "filters" field in contract only supports: date_from, date_to, status, tags
    # It does NOT support arbitrary metadata fields like "author"
    if "filters" in retrieve_payload:
        # If filters is present, it should not contain "author"
        assert "author" not in retrieve_payload.get("filters", {}), \
            "Author should be in metadata_filters, not filters"


@pytest.mark.asyncio
async def test_dataset_includes_graph_relationship_test_cases(golden_fact_check):
    """Test that S2 dataset includes 3 graph relationship test cases (code review requirement)."""
    # Verify we have 6 test cases total
    assert len(golden_fact_check.test_dataset) == 6, \
        f"Expected 6 test cases, got {len(golden_fact_check.test_dataset)}"

    # Count test types
    semantic_tests = [t for t in golden_fact_check.test_dataset if t.get("test_type") == "semantic_search"]
    graph_tests = [t for t in golden_fact_check.test_dataset if t.get("test_type") == "graph_relationship"]

    assert len(semantic_tests) == 3, f"Expected 3 semantic tests, got {len(semantic_tests)}"
    assert len(graph_tests) == 3, f"Expected 3 graph tests, got {len(graph_tests)}"

    # Verify graph test cases contain expected content
    graph_test_contents = [t["kv"] for t in graph_tests]

    # Test case 1: Project tech stack
    tech_stack_test = next((t for t in graph_test_contents if "Veris Memory" in str(t)), None)
    assert tech_stack_test is not None, "Missing project tech stack graph test case"
    assert "Python" in str(tech_stack_test) or "FastAPI" in str(tech_stack_test), \
        "Tech stack test should reference Python/FastAPI/Neo4j"

    # Test case 2: Feature dependencies
    dependencies_test = next((t for t in graph_test_contents if "semantic search" in str(t)), None)
    assert dependencies_test is not None, "Missing feature dependencies graph test case"
    assert "Qdrant" in str(dependencies_test) or "vector" in str(dependencies_test), \
        "Dependencies test should reference Qdrant/vector database"

    # Test case 3: Component relationships
    component_test = next((t for t in graph_test_contents if "monitoring" in str(t)), None)
    assert component_test is not None, "Missing component relationships graph test case"
    assert "Sentinel" in str(component_test) or "metrics" in str(component_test), \
        "Component test should reference Sentinel/metrics/Neo4j"


@pytest.mark.asyncio
async def test_graph_relationship_queries_properly_formed(golden_fact_check):
    """Test that graph relationship test cases have proper structure."""
    graph_tests = [t for t in golden_fact_check.test_dataset if t.get("test_type") == "graph_relationship"]

    for i, test in enumerate(graph_tests):
        # Each test should have kv, questions, and expect_contains
        assert "kv" in test, f"Graph test {i} missing 'kv' field"
        assert "questions" in test, f"Graph test {i} missing 'questions' field"
        assert "expect_contains" in test, f"Graph test {i} missing 'expect_contains' field"
        assert "test_type" in test, f"Graph test {i} missing 'test_type' field"

        # Each test should have exactly 2 questions
        assert len(test["questions"]) == 2, \
            f"Graph test {i} should have 2 questions, got {len(test['questions'])}"

        # Questions should be non-empty strings
        for q in test["questions"]:
            assert isinstance(q, str) and len(q) > 0, \
                f"Graph test {i} has invalid question: {q}"

        # expect_contains should be non-empty string
        assert isinstance(test["expect_contains"], str) and len(test["expect_contains"]) > 0, \
            f"Graph test {i} has invalid expect_contains: {test['expect_contains']}"


@pytest.mark.asyncio
async def test_total_query_count_is_12(golden_fact_check):
    """Test that S2 generates exactly 12 queries (6 test cases × 2 questions each)."""
    total_queries = sum(len(test["questions"]) for test in golden_fact_check.test_dataset)

    assert total_queries == 12, \
        f"Expected 12 total queries (6 test cases × 2 questions), got {total_queries}"

    # Verify breakdown
    semantic_queries = sum(
        len(test["questions"])
        for test in golden_fact_check.test_dataset
        if test.get("test_type") == "semantic_search"
    )
    graph_queries = sum(
        len(test["questions"])
        for test in golden_fact_check.test_dataset
        if test.get("test_type") == "graph_relationship"
    )

    assert semantic_queries == 6, f"Expected 6 semantic queries (3 × 2), got {semantic_queries}"
    assert graph_queries == 6, f"Expected 6 graph queries (3 × 2), got {graph_queries}"


# ============================================================================
# PR #399: Cleanup Tests - Prevent database bloat from sentinel test data
# PR #403: Updated to test multi-backend cleanup (Neo4j + Qdrant)
# ============================================================================

@pytest.mark.asyncio
async def test_cleanup_fact_deletes_context(golden_fact_check):
    """Test that _cleanup_fact sends cleanup request via internal endpoint (PR #403)."""
    mock_session = AsyncMock()
    context_id = "ctx_test_cleanup_123"
    captured_payload = None

    async def capture_cleanup_call(session, method, url, data, expected_status, timeout):
        nonlocal captured_payload
        captured_payload = data
        # PR #403: Return new sentinel cleanup response format
        return True, "Success", 5.0, {
            "success": True,
            "deleted_from": ["neo4j", "qdrant"],
            "message": "Sentinel test data cleaned from: neo4j, qdrant"
        }

    with patch.object(golden_fact_check, 'test_api_call', side_effect=capture_cleanup_call):
        result = await golden_fact_check._cleanup_fact(mock_session, context_id)

    # Verify cleanup was attempted
    assert result["success"] is True
    assert result["context_id"] == context_id
    assert "deleted_from" in result
    assert "neo4j" in result["deleted_from"]
    assert "qdrant" in result["deleted_from"]

    # Verify the payload structure for internal cleanup endpoint
    assert captured_payload is not None
    assert "context_id" in captured_payload
    assert captured_payload["context_id"] == context_id
    assert "sentinel_key" in captured_payload


@pytest.mark.asyncio
async def test_cleanup_fact_handles_no_context_id(golden_fact_check):
    """Test that _cleanup_fact handles missing context_id gracefully (PR #399)."""
    mock_session = AsyncMock()

    result = await golden_fact_check._cleanup_fact(mock_session, None)

    assert result["success"] is False
    assert "No context_id" in result["message"]


@pytest.mark.asyncio
async def test_cleanup_fact_rejects_invalid_context_id_format(golden_fact_check):
    """Test that _cleanup_fact rejects context IDs with injection characters (PR #399 security)."""
    mock_session = AsyncMock()

    # Test various injection attempts
    injection_attempts = [
        "'; DROP TABLE contexts; --",
        "test' OR '1'='1",
        "ctx_123; MATCH (n) DETACH DELETE n",
        "ctx_<script>alert(1)</script>",
        "ctx_123\n RETURN 1",
        "ctx with spaces",
    ]

    for malicious_id in injection_attempts:
        result = await golden_fact_check._cleanup_fact(mock_session, malicious_id)
        assert result["success"] is False, f"Should reject: {malicious_id}"
        assert "Invalid context_id format" in result["message"], f"Wrong message for: {malicious_id}"


@pytest.mark.asyncio
async def test_cleanup_fact_accepts_valid_context_id_formats(golden_fact_check):
    """Test that _cleanup_fact accepts valid context ID formats (PR #399 security)."""
    mock_session = AsyncMock()

    # Valid context ID formats
    valid_ids = [
        "ctx_abc123",
        "58c677bc-44a6-4597-890e-c36f672d9325",  # UUID
        "context_12345",
        "ABC123",
        "a1b2c3d4e5f6",
    ]

    async def mock_api_call(session, method, url, data, expected_status, timeout):
        # PR #403: Return new sentinel cleanup response format
        return True, "Success", 5.0, {
            "success": True,
            "deleted_from": ["neo4j", "qdrant"],
            "message": "Cleaned"
        }

    with patch.object(golden_fact_check, 'test_api_call', side_effect=mock_api_call):
        for valid_id in valid_ids:
            result = await golden_fact_check._cleanup_fact(mock_session, valid_id)
            assert result["success"] is True, f"Should accept: {valid_id}"


@pytest.mark.asyncio
async def test_cleanup_fact_handles_api_failure(golden_fact_check):
    """Test that _cleanup_fact handles API failures gracefully (PR #399)."""
    mock_session = AsyncMock()
    context_id = "ctx_test_failure"

    async def mock_api_failure(session, method, url, data, expected_status, timeout):
        return False, "Connection error", 5.0, None

    with patch.object(golden_fact_check, 'test_api_call', side_effect=mock_api_failure):
        result = await golden_fact_check._cleanup_fact(mock_session, context_id)

    assert result["success"] is False
    assert result["context_id"] == context_id


@pytest.mark.asyncio
async def test_run_single_test_calls_cleanup_after_recall(golden_fact_check):
    """Test that _run_single_test calls cleanup after testing (PR #399)."""
    mock_session = AsyncMock()
    test_case = {
        "kv": {"name": "Test"},
        "questions": ["What's the test?"],
        "expect_contains": "Test",
        "test_type": "semantic_search"
    }

    # Track which methods were called
    calls = {"store": 0, "recall": 0, "cleanup": 0}
    stored_context_id = "ctx_stored_123"

    async def mock_store(session, fact_data, user_id):
        calls["store"] += 1
        return {
            "success": True,
            "response": {"id": stored_context_id}
        }

    async def mock_recall(session, question, expected, user_id):
        calls["recall"] += 1
        return {"success": True}

    async def mock_cleanup(session, context_id):
        calls["cleanup"] += 1
        assert context_id == stored_context_id, "Cleanup should use stored context_id"
        return {"success": True, "deleted_count": 1}

    with patch.object(golden_fact_check, '_store_fact', side_effect=mock_store), \
         patch.object(golden_fact_check, '_test_recall', side_effect=mock_recall), \
         patch.object(golden_fact_check, '_cleanup_fact', side_effect=mock_cleanup):
        result = await golden_fact_check._run_single_test(mock_session, test_case, 0)

    # Verify cleanup was called
    assert calls["store"] == 1, "Store should be called once"
    assert calls["recall"] == 1, "Recall should be called for each question"
    assert calls["cleanup"] == 1, "Cleanup should be called once after recall tests"

    # Verify result includes cleanup info
    assert "cleanup_result" in result
    assert result["cleanup_result"]["success"] is True


@pytest.mark.asyncio
async def test_run_single_test_includes_cleanup_in_result(golden_fact_check):
    """Test that _run_single_test includes cleanup result in output (PR #399)."""
    mock_session = AsyncMock()
    test_case = {
        "kv": {"name": "Test"},
        "questions": ["Q1", "Q2"],
        "expect_contains": "Test",
        "test_type": "semantic_search"
    }

    async def mock_api_call(session, method, url, data, expected_status, timeout):
        if "store_context" in url:
            return True, "Success", 10.0, {"id": "ctx_cleanup_test"}
        elif "retrieve_context" in url:
            return True, "Success", 15.0, {
                "results": [{"content": {"name": "Test"}}]
            }
        elif "query_graph" in url:
            return True, "Success", 5.0, {"results": [{"deleted": 1}]}
        return False, "Unknown URL", 0, None

    with patch.object(golden_fact_check, 'test_api_call', side_effect=mock_api_call):
        result = await golden_fact_check._run_single_test(mock_session, test_case, 0)

    # Result should include cleanup info
    assert "cleanup_result" in result
    assert result["cleanup_result"] is not None


@pytest.mark.asyncio
async def test_cleanup_not_called_when_store_fails(golden_fact_check):
    """Test that cleanup is not called when store fails (no context to clean up)."""
    mock_session = AsyncMock()
    test_case = {
        "kv": {"name": "Test"},
        "questions": ["What?"],
        "expect_contains": "Test",
        "test_type": "semantic_search"
    }

    cleanup_called = False

    async def mock_store_failure(session, fact_data, user_id):
        return {"success": False, "message": "Store failed"}

    async def mock_cleanup(session, context_id):
        nonlocal cleanup_called
        cleanup_called = True
        return {"success": True}

    with patch.object(golden_fact_check, '_store_fact', side_effect=mock_store_failure), \
         patch.object(golden_fact_check, '_cleanup_fact', side_effect=mock_cleanup):
        result = await golden_fact_check._run_single_test(mock_session, test_case, 0)

    # Cleanup should NOT be called since store failed
    assert cleanup_called is False, "Cleanup should not be called when store fails"
    assert result["store_success"] is False
    assert result["cleanup_result"] is None
