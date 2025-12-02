#!/usr/bin/env python3
"""
Simplified Context Store MCP Server.

This module implements a basic MCP server that can be extended with
full storage integration later.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Sequence

# MCP SDK imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, Resource, TextContent, Tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("context-store")

# Simple in-memory storage for demonstration
context_storage: Dict[str, Dict[str, Any]] = {}
next_id = 1


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="context://health",
            name="Health Status",
            description="Server health status",
            mimeType="application/json",
        ),
        Resource(
            uri="context://storage",
            name="Storage Info",
            description="Storage system information",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content."""
    if uri == "context://health":
        health_data = {
            "status": "healthy",
            "server": "context-store-mcp",
            "version": "1.0.0",
            "timestamp": asyncio.get_event_loop().time(),
        }
        return json.dumps(health_data, indent=2)

    elif uri == "context://storage":
        storage_data = {
            "stored_contexts": len(context_storage),
            "storage_type": "in-memory",
            "contexts": list(context_storage.keys()),
        }
        return json.dumps(storage_data, indent=2)

    else:
        raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="store_context",
            description="Store context data in the context store",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "object",
                        "description": "Context content to store",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["design", "decision", "trace", "sprint", "log"],
                        "description": "Type of context",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata",
                    },
                },
                "required": ["content", "type"],
            },
        ),
        Tool(
            name="retrieve_context",
            description="Retrieve stored context data",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "type": {
                        "type": "string",
                        "description": "Context type filter (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Maximum results to return",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_contexts",
            description="List all stored contexts",
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Filter by context type (optional)",
                    }
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: Dict[str, Any]
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    if name == "store_context":
        result = await handle_store_context(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "retrieve_context":
        result = await handle_retrieve_context(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "list_contexts":
        result = await handle_list_contexts(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_store_context(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Store context data."""
    global next_id

    try:
        content = arguments["content"]
        context_type = arguments["type"]
        metadata = arguments.get("metadata", {})

        # Generate ID
        context_id = f"ctx_{next_id:06d}"
        next_id += 1

        # Store context
        context_storage[context_id] = {
            "id": context_id,
            "content": content,
            "type": context_type,
            "metadata": metadata,
            "created_at": asyncio.get_event_loop().time(),
        }

        logger.info(f"Stored context {context_id} of type {context_type}")

        return {
            "success": True,
            "id": context_id,
            "type": context_type,
            "message": f"Context stored successfully with ID {context_id}",
        }

    except Exception as e:
        logger.error(f"Error storing context: {e}")
        return {"success": False, "error": str(e), "message": "Failed to store context"}


async def handle_retrieve_context(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve context data."""
    try:
        query = arguments["query"].lower()
        context_type_filter = arguments.get("type")
        limit = arguments.get("limit", 10)
        sort_by = arguments.get("sort_by", "timestamp")  # Default to timestamp
        
        # Validate sort_by parameter
        if sort_by not in ["timestamp", "relevance"]:
            return {
                "success": False,
                "results": [],
                "message": f"Invalid sort_by value: '{sort_by}'. Must be 'timestamp' or 'relevance'",
                "error_type": "invalid_parameter"
            }

        results = []

        # Simple search through stored contexts
        for ctx_id, ctx_data in context_storage.items():
            # Type filter
            if context_type_filter and ctx_data["type"] != context_type_filter:
                continue

            # Simple text search in content
            content_str = json.dumps(ctx_data["content"]).lower()
            if query in content_str or query in ctx_id.lower():
                results.append(
                    {
                        "id": ctx_id,
                        "type": ctx_data["type"],
                        "content": ctx_data["content"],
                        "metadata": ctx_data["metadata"],
                        "relevance": 1.0 if query in ctx_id.lower() else 0.5,
                        "created_at": ctx_data.get("metadata", {}).get("created_at", "")
                    }
                )

        # Sort based on sort_by parameter
        if sort_by == "timestamp":
            results.sort(key=lambda x: x.get("created_at", "") or "", reverse=True)
        elif sort_by == "relevance":
            results.sort(key=lambda x: x["relevance"], reverse=True)
        results = results[:limit]

        logger.info(f"Retrieved {len(results)} contexts for query: {query}")

        return {
            "success": True,
            "query": arguments["query"],
            "results": results,
            "total_found": len(results),
            "message": f"Found {len(results)} matching contexts",
        }

    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "message": "Failed to retrieve context",
        }


async def handle_list_contexts(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """List all contexts."""
    try:
        context_type_filter = arguments.get("type")

        contexts = []
        for ctx_id, ctx_data in context_storage.items():
            if context_type_filter and ctx_data["type"] != context_type_filter:
                continue

            contexts.append(
                {
                    "id": ctx_id,
                    "type": ctx_data["type"],
                    "created_at": ctx_data["created_at"],
                    "has_metadata": bool(ctx_data["metadata"]),
                }
            )

        # Sort by creation time (newest first)
        contexts.sort(key=lambda x: x["created_at"], reverse=True)

        return {
            "success": True,
            "contexts": contexts,
            "total_count": len(contexts),
            "filter_type": context_type_filter,
            "message": f"Listed {len(contexts)} contexts",
        }

    except Exception as e:
        logger.error(f"Error listing contexts: {e}")
        return {
            "success": False,
            "error": str(e),
            "contexts": [],
            "message": "Failed to list contexts",
        }


async def main():
    """Main server entry point."""
    logger.info("Starting Context Store MCP Server...")

    try:
        # Run the server using stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="context-store",
                    server_version="1.0.0",
                    capabilities={}
                ),
            )
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
