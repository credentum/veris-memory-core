#!/usr/bin/env python3
"""
Tests for the Simple Context Store MCP Server.
"""

import asyncio
import json

import pytest

from src.mcp_server.simple_server import (
    context_storage,
    handle_list_contexts,
    handle_retrieve_context,
    handle_store_context,
)


class TestSimpleMCPServer:
    """Test cases for the simple MCP server."""

    def setup_method(self):
        """Reset storage before each test."""
        context_storage.clear()
        # Reset the ID counter
        from src.mcp_server import simple_server

        simple_server.next_id = 1

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test resource listing."""
        # Import the functions directly since server handlers are internal
        from src.mcp_server.simple_server import list_resources

        resources = await list_resources()

        assert len(resources) >= 2
        resource_uris = [str(r.uri) for r in resources]
        assert "context://health" in resource_uris
        assert "context://storage" in resource_uris

    @pytest.mark.asyncio
    async def test_read_health_resource(self):
        """Test reading health resource."""
        from src.mcp_server.simple_server import read_resource

        health_data = await read_resource("context://health")
        health_json = json.loads(health_data)

        assert health_json["status"] == "healthy"
        assert health_json["server"] == "context-store-mcp"
        assert health_json["version"] == "1.0.0"
        assert "timestamp" in health_json

    @pytest.mark.asyncio
    async def test_read_storage_resource(self):
        """Test reading storage resource."""
        from src.mcp_server.simple_server import read_resource

        storage_data = await read_resource("context://storage")
        storage_json = json.loads(storage_data)

        assert "stored_contexts" in storage_json
        assert storage_json["storage_type"] == "in-memory"
        assert "contexts" in storage_json

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test tool listing."""
        from src.mcp_server.simple_server import list_tools

        tools = await list_tools()

        assert len(tools) >= 3
        tool_names = [t.name for t in tools]
        assert "store_context" in tool_names
        assert "retrieve_context" in tool_names
        assert "list_contexts" in tool_names

    @pytest.mark.asyncio
    async def test_store_context(self):
        """Test storing context."""
        arguments = {
            "content": {"title": "Test Context", "description": "Test description"},
            "type": "design",
            "metadata": {"author": "test_user"},
        }

        result = await handle_store_context(arguments)

        assert result["success"] is True
        assert "id" in result
        assert result["id"].startswith("ctx_")
        assert result["type"] == "design"
        assert "message" in result

        # Verify it's stored
        assert len(context_storage) == 1
        stored_context = list(context_storage.values())[0]
        assert stored_context["type"] == "design"
        assert stored_context["content"]["title"] == "Test Context"

    @pytest.mark.asyncio
    async def test_store_multiple_contexts(self):
        """Test storing multiple contexts."""
        contexts = [
            {"content": {"title": "Context 1"}, "type": "design"},
            {"content": {"title": "Context 2"}, "type": "decision"},
            {"content": {"title": "Context 3"}, "type": "design"},
        ]

        stored_ids = []
        for ctx in contexts:
            result = await handle_store_context(ctx)
            assert result["success"] is True
            stored_ids.append(result["id"])

        assert len(context_storage) == 3
        assert len(set(stored_ids)) == 3  # All IDs are unique

    @pytest.mark.asyncio
    async def test_retrieve_context(self):
        """Test retrieving context."""
        # Store a context first
        store_args = {
            "content": {
                "title": "Searchable Context",
                "description": "This is searchable",
            },
            "type": "design",
        }
        store_result = await handle_store_context(store_args)
        assert store_result["success"] is True

        # Now search for it
        search_args = {"query": "Searchable", "limit": 10}

        result = await handle_retrieve_context(search_args)

        assert result["success"] is True
        assert result["query"] == "Searchable"
        assert len(result["results"]) == 1
        assert result["results"][0]["content"]["title"] == "Searchable Context"
        assert result["total_found"] == 1

    @pytest.mark.asyncio
    async def test_retrieve_context_with_type_filter(self):
        """Test retrieving context with type filter."""
        # Store contexts of different types
        contexts = [
            {"content": {"title": "Design Doc"}, "type": "design"},
            {"content": {"title": "Design Decision"}, "type": "decision"},
            {"content": {"title": "Design Plan"}, "type": "design"},
        ]

        for ctx in contexts:
            await handle_store_context(ctx)

        # Search for design contexts only
        search_args = {"query": "Design", "type": "design", "limit": 10}

        result = await handle_retrieve_context(search_args)

        assert result["success"] is True
        assert len(result["results"]) == 2  # Only design contexts
        for res in result["results"]:
            assert res["type"] == "design"

    @pytest.mark.asyncio
    async def test_retrieve_context_no_matches(self):
        """Test retrieving context with no matches."""
        search_args = {"query": "NonexistentTerm", "limit": 10}

        result = await handle_retrieve_context(search_args)

        assert result["success"] is True
        assert len(result["results"]) == 0
        assert result["total_found"] == 0

    @pytest.mark.asyncio
    async def test_list_contexts_empty(self):
        """Test listing contexts when storage is empty."""
        result = await handle_list_contexts({})

        assert result["success"] is True
        assert len(result["contexts"]) == 0
        assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_list_contexts_with_data(self):
        """Test listing contexts with stored data."""
        # Store some contexts
        contexts = [
            {"content": {"title": "Context 1"}, "type": "design"},
            {"content": {"title": "Context 2"}, "type": "decision"},
        ]

        for ctx in contexts:
            await handle_store_context(ctx)

        result = await handle_list_contexts({})

        assert result["success"] is True
        assert len(result["contexts"]) == 2
        assert result["total_count"] == 2

        # Check context info
        context_ids = [ctx["id"] for ctx in result["contexts"]]
        context_types = [ctx["type"] for ctx in result["contexts"]]

        assert len(set(context_ids)) == 2  # Unique IDs
        assert "design" in context_types
        assert "decision" in context_types

    @pytest.mark.asyncio
    async def test_list_contexts_with_type_filter(self):
        """Test listing contexts with type filter."""
        # Store contexts of different types
        contexts = [
            {"content": {"title": "Design 1"}, "type": "design"},
            {"content": {"title": "Decision 1"}, "type": "decision"},
            {"content": {"title": "Design 2"}, "type": "design"},
        ]

        for ctx in contexts:
            await handle_store_context(ctx)

        # Filter by design type
        result = await handle_list_contexts({"type": "design"})

        assert result["success"] is True
        assert len(result["contexts"]) == 2
        assert result["filter_type"] == "design"

        for ctx in result["contexts"]:
            assert ctx["type"] == "design"

    @pytest.mark.asyncio
    async def test_store_context_error_handling(self):
        """Test error handling in store_context."""
        # Test with missing required field
        arguments = {
            "content": {"title": "Test"},
            # Missing "type" field
        }

        result = await handle_store_context(arguments)

        assert result["success"] is False
        assert "error" in result
        assert "message" in result

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete store -> retrieve -> list workflow."""
        # Store a context
        store_args = {
            "content": {"title": "Workflow Test", "data": "test data"},
            "type": "design",
            "metadata": {"author": "test", "version": "1.0"},
        }

        store_result = await handle_store_context(store_args)
        assert store_result["success"] is True
        context_id = store_result["id"]

        # Retrieve it
        retrieve_args = {"query": "Workflow Test", "limit": 5}

        retrieve_result = await handle_retrieve_context(retrieve_args)
        assert retrieve_result["success"] is True
        assert len(retrieve_result["results"]) == 1
        assert retrieve_result["results"][0]["id"] == context_id

        # List all contexts
        list_result = await handle_list_contexts({})
        assert list_result["success"] is True
        assert len(list_result["contexts"]) == 1
        assert list_result["contexts"][0]["id"] == context_id

    @pytest.mark.asyncio
    async def test_context_ordering(self):
        """Test that contexts are ordered by creation time."""

        # Store contexts with small delays to ensure different timestamps
        contexts = [
            {"content": {"title": "First"}, "type": "design"},
            {"content": {"title": "Second"}, "type": "design"},
            {"content": {"title": "Third"}, "type": "design"},
        ]

        for i, ctx in enumerate(contexts):
            await handle_store_context(ctx)
            if i < len(contexts) - 1:  # Don't wait after the last one
                await asyncio.sleep(0.01)  # Small delay

        # List contexts (should be newest first)
        result = await handle_list_contexts({})

        assert result["success"] is True
        assert len(result["contexts"]) == 3

        # Check ordering (newest first)
        timestamps = [ctx["created_at"] for ctx in result["contexts"]]
        assert timestamps == sorted(timestamps, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__])
