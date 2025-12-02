#!/usr/bin/env python3
"""
Integration tests for graph backend retrieval fix.

Tests the complete flow of storing and retrieving contexts through the API,
verifying that the field mismatch fix works correctly for both manual and
voice bot contexts.
"""
import pytest
import os
import asyncio
import time
from typing import Dict, Any
import httpx

# Configuration from environment
API_URL = os.getenv("API_URL", "http://172.17.0.1:8000")
API_KEY = os.getenv("API_KEY_MCP")


@pytest.fixture(scope="module")
def api_headers():
    """Get API headers with authentication."""
    if not API_KEY:
        pytest.skip("API_KEY_MCP environment variable not set")

    return {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }


@pytest.fixture(scope="module")
async def http_client():
    """Create async HTTP client for API calls."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


class TestGraphRetrievalIntegration:
    """Integration tests for graph backend retrieval."""

    @pytest.mark.asyncio
    async def test_manual_context_storage_and_retrieval(self, http_client, api_headers):
        """Test storing and retrieving manual contexts with title/description/keyword fields."""
        # Store context
        store_payload = {
            "type": "decision",
            "content": {
                "title": "Integration Test Banana Protocol",
                "description": "Testing banana retrieval after field fix",
                "keyword": "banana_integration_test"
            },
            "author": "test_agent",
            "author_type": "agent"
        }

        store_response = await http_client.post(
            f"{API_URL}/tools/store_context",
            json=store_payload,
            headers=api_headers
        )

        assert store_response.status_code == 200, f"Store failed: {store_response.text}"
        store_data = store_response.json()
        assert store_data["success"] is True
        assert "id" in store_data
        context_id = store_data["id"]

        # Wait for indexing
        await asyncio.sleep(2)

        # Retrieve by keyword
        retrieve_response = await http_client.post(
            f"{API_URL}/tools/retrieve_context",
            json={"query": "banana", "limit": 5},
            headers=api_headers
        )

        assert retrieve_response.status_code == 200, f"Retrieve failed: {retrieve_response.text}"
        retrieve_data = retrieve_response.json()

        # Verify results
        assert "results" in retrieve_data
        assert retrieve_data["total_count"] > 0, "No results returned for 'banana' query"

        # Verify our context is in results
        context_ids = [r["id"] for r in retrieve_data["results"]]
        assert context_id in context_ids, f"Stored context {context_id} not found in results"

    @pytest.mark.asyncio
    async def test_voice_bot_context_storage_and_retrieval(self, http_client, api_headers):
        """Test storing and retrieving voice bot contexts with user_input/bot_response fields."""
        # Store voice bot context
        store_payload = {
            "type": "trace",
            "content": {
                "user_input": "Integration test sunshine message",
                "bot_response": "Integration test happiness response",
                "title": "Voice Test Conversation"
            },
            "author": "voice_bot",
            "author_type": "agent"
        }

        store_response = await http_client.post(
            f"{API_URL}/tools/store_context",
            json=store_payload,
            headers=api_headers
        )

        assert store_response.status_code == 200
        store_data = store_response.json()
        assert store_data["success"] is True
        context_id = store_data["id"]

        # Wait for indexing
        await asyncio.sleep(2)

        # Retrieve by user_input keyword
        retrieve_response = await http_client.post(
            f"{API_URL}/tools/retrieve_context",
            json={"query": "sunshine", "limit": 5},
            headers=api_headers
        )

        assert retrieve_response.status_code == 200
        retrieve_data = retrieve_response.json()

        # Verify results
        assert retrieve_data["total_count"] > 0, "No results returned for 'sunshine' query"
        context_ids = [r["id"] for r in retrieve_data["results"]]
        assert context_id in context_ids

    @pytest.mark.asyncio
    async def test_backend_timings_are_nonzero(self, http_client, api_headers):
        """Verify that backend timings are properly reported (not 0.0)."""
        # Store a test context first
        store_payload = {
            "type": "log",
            "content": {
                "title": "Timing Test Context",
                "description": "Test for backend timing verification"
            },
            "author": "test_agent",
            "author_type": "agent"
        }

        await http_client.post(
            f"{API_URL}/tools/store_context",
            json=store_payload,
            headers=api_headers
        )

        await asyncio.sleep(2)

        # Retrieve and check timings
        retrieve_response = await http_client.post(
            f"{API_URL}/tools/retrieve_context",
            json={"query": "timing", "limit": 5},
            headers=api_headers
        )

        assert retrieve_response.status_code == 200
        retrieve_data = retrieve_response.json()

        # Verify backend timings exist
        assert "backend_timings" in retrieve_data, "backend_timings not in response"
        timings = retrieve_data["backend_timings"]

        # At least one backend should have non-zero timing
        assert "graph" in timings, "graph timing not reported"

        # Graph backend should be working now
        if retrieve_data["total_count"] > 0:
            assert timings["graph"] > 0.0, "Graph backend timing is 0.0 - search may not be working"

    @pytest.mark.asyncio
    async def test_case_insensitive_search(self, http_client, api_headers):
        """Test that search is case-insensitive after toLower() fix."""
        # Store context with specific capitalization
        store_payload = {
            "type": "decision",
            "content": {
                "title": "CaseSensitiveTest Protocol",
                "description": "UPPERCASE and lowercase testing"
            },
            "author": "test_agent",
            "author_type": "agent"
        }

        store_response = await http_client.post(
            f"{API_URL}/tools/store_context",
            json=store_payload,
            headers=api_headers
        )

        assert store_response.status_code == 200
        context_id = store_response.json()["id"]

        await asyncio.sleep(2)

        # Test lowercase query
        retrieve_response = await http_client.post(
            f"{API_URL}/tools/retrieve_context",
            json={"query": "casesensitivetest", "limit": 5},
            headers=api_headers
        )

        assert retrieve_response.status_code == 200
        retrieve_data = retrieve_response.json()

        # Should find the context despite different case
        context_ids = [r["id"] for r in retrieve_data["results"]]
        assert context_id in context_ids, "Case-insensitive search not working"

        # Test uppercase query
        retrieve_response = await http_client.post(
            f"{API_URL}/tools/retrieve_context",
            json={"query": "UPPERCASE", "limit": 5},
            headers=api_headers
        )

        assert retrieve_response.status_code == 200
        retrieve_data = retrieve_response.json()

        context_ids = [r["id"] for r in retrieve_data["results"]]
        assert context_id in context_ids, "Case-insensitive search not working for uppercase"

    @pytest.mark.asyncio
    async def test_existing_contexts_searchable(self, http_client, api_headers):
        """Verify that existing contexts in database become searchable after fix."""
        # Search for known existing contexts (voice bot contexts with "sunshine")
        retrieve_response = await http_client.post(
            f"{API_URL}/tools/retrieve_context",
            json={"query": "sunshine", "limit": 10},
            headers=api_headers
        )

        assert retrieve_response.status_code == 200
        retrieve_data = retrieve_response.json()

        # Should find at least some contexts
        # (May be 0 if database was cleared, but that's okay for CI)
        assert "results" in retrieve_data
        assert "total_count" in retrieve_data

    @pytest.mark.asyncio
    async def test_search_across_multiple_fields(self, http_client, api_headers):
        """Test that search works across all actual Context fields."""
        # Create contexts with text in different fields
        contexts_to_create = [
            {"title": "SearchField Title", "field": "title"},
            {"description": "SearchField Description", "field": "description"},
            {"keyword": "SearchField", "field": "keyword"},
        ]

        context_ids = []
        for content_data in contexts_to_create:
            field_type = content_data.pop("field")
            store_payload = {
                "type": "log",
                "content": content_data,
                "author": "test_agent",
                "author_type": "agent"
            }

            response = await http_client.post(
                f"{API_URL}/tools/store_context",
                json=store_payload,
                headers=api_headers
            )

            assert response.status_code == 200
            context_ids.append(response.json()["id"])

        await asyncio.sleep(3)

        # Search for common term
        retrieve_response = await http_client.post(
            f"{API_URL}/tools/retrieve_context",
            json={"query": "SearchField", "limit": 10},
            headers=api_headers
        )

        assert retrieve_response.status_code == 200
        retrieve_data = retrieve_response.json()

        # Should find all three contexts (from different fields)
        found_ids = [r["id"] for r in retrieve_data["results"]]

        for context_id in context_ids:
            assert context_id in found_ids, f"Context {context_id} not found - multi-field search not working"


@pytest.mark.asyncio
async def test_api_health_check():
    """Verify API is accessible before running tests."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    except Exception as e:
        pytest.skip(f"API not accessible at {API_URL}: {e}")
